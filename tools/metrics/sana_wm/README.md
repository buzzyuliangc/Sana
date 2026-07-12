# SANA-WM Inference Speed Optimization

Latency optimization of SANA-WM inference with hard quality gates, plus the
benchmark harness used to prove every claim. All work lives on the
`wm-speed-opt` branch; the complete result-by-result log and forward plan is
in [`SPEED_ROADMAP.md`](SPEED_ROADMAP.md).

## Headline result (bidirectional pipeline)

**1.71× faster generation (920 s → 538 s per 60-second 720p scene) with
camera adherence 10–28% better than the paper's published configuration and
visual quality at parity** — validated on the full official 80-scene
benchmark, both splits, against the paper's Table-2 baselines
(arXiv 2605.15178).

| Metric (160 scenes) | Official (paper) | This work (stack B) |
|---|---|---|
| RotErr simple / hard | 4.50° / 8.34° | **3.59° / 6.02°** |
| TransErr simple / hard | 1.39 / 1.39 | **1.20 / 1.25** |
| CamMC simple / hard | 1.41 / 1.44 | **1.22 / 1.28** |
| VBench simple / hard | 80.62 / 81.89 | 79.75 / 81.75 |
| Wall clock / scene | ~920 s | **538 s** |

**Stack B** = `--step 40 --cfg_truncate_ratio 0.5 --cache_camera_geometry`.

## What was adopted, what was rejected

| Verdict | Optimization | Speed | Quality outcome |
|---|---|---|---|
| ✅ | CFG truncation 0.5 (`--cfg_truncate_ratio`) | 1.26× | **improves** camera adherence ~9% — late-step guidance fights the camera branch |
| ✅ | 40 steps (in the stack) | 1.38× | passes in combination with truncation |
| ✅ | Camera-geometry caching (`--cache_camera_geometry`) | 1.013× | bit-exact (stage-1 latents max\|Δ\| = 0) |
| ✅ | Async MP4 writer, streaming (`--async_writer`, default on) | ~15% expected | lossless (identical bytes to ffmpeg) |
| ❌ | fp8 W8A8 per-tensor (`SANA_WM_STAGE1_QUANT=fp8delayed`) | 0% (null result) | conversion verified active; cast/amax overhead cancels GEMM gains — retest needs CUDA ≥ 12.9 (block scaling) or Blackwell (fp4) |
| ⏸ | `--compile` (per-block torch.compile) | 1.20×/step, 1.04× e2e | serving deployments only (per-process warmup; bf16 trajectory drift) |
| ❌ | steps 30 / 20, cfg 0.3 | 1.42–2.23× | pose gates fail (+22–44% RotErr) |
| ❌ | Flow-DPM-Solver (any steps) | up to 2.15× | sampler lacks LTXFlowEuler's first-frame anchoring → conditioning erodes (RotErr up to 21°) |
| ❌ | SageAttention (`SANA_WM_SAGE_ATTENTION=1`) | +6.5% | −3.9 VBench: genuine INT8 visual cost |
| ❌ | Step caching (`--step_cache_interval 2`) | 2.24× | −3.9 VBench, RotErr +7.2% |

The pattern: **removing redundancy passes; approximating the computation
fails.** Stack B sits at the redundancy-free frontier of this model.

## Streaming pipeline findings (S0)

Measured on H100 (RunPod), 321/961-frame clips: steady-state 0.66× realtime
writing MP4, 0.79× with output discarded → **encoding sits on the critical
path (~15%)**, fixed by the async writer. Per-stage GPU share: stage-1 58%,
refiner 29%, decode 13% — stage-1 GPU work alone exceeds the realtime
budget, so kernel work (not pipelining) is the path to 1×. The gap to the
published 1.09× is real host/config overhead, not warmup. Kernel plan
(CUDA graphs → fused GDN in Triton/CuTe DSL → ring-buffer KV attention) is
scoped with effort/gain estimates in the roadmap.

## New flags & tools reference

Bidirectional (`inference_video_scripts/wm/inference_sana_wm.py`), all
default-off unless noted:

- `--timing_json PATH` — per-stage wall/CUDA seconds, per-step DiT
  latencies, per-stage peak VRAM, environment block
- `--save_stage1_latents PATH` — pre-refiner latents for equivalence checks
- `--cache_camera_geometry` — UCPE raymats / RoPE / Plücker computed once
  per scene (bit-exact; flow_euler_ltx/flow_euler only)
- `--cfg_truncate_ratio R` — CFG for the first `ceil(R·steps)` steps only
- `--step_cache_interval N` — FORA-style mid-trajectory velocity reuse
- `--refiner_sigmas` — refiner Euler schedule override
- `--compile` — per-block torch.compile
- env `SANA_WM_SAGE_ATTENTION=1` — SageAttention on the softmax blocks
- env `SANA_WM_STAGE1_QUANT=fp8delayed` — Hopper/CUDA-12.8-compatible fp8
  (Float8BlockScaling needs CUDA ≥ 12.9; NVFP4 needs Blackwell)

Streaming (`inference_sana_wm_streaming.py`): `--async_writer` /
`--no-async_writer` (default on).

## Reproducing the benchmark

```bash
# 1. Pin a smoke set (2 scenes per category)
python tools/metrics/sana_wm/speed_bench.py pick-smoke \
  --manifest data/SANA-WM-Bench/benchmark_v2_smooth_60s/sanawm_export_v2/run_manifest.jsonl

# 2. Baseline + variant runs (one subprocess per scene, official settings)
python tools/metrics/sana_wm/speed_bench.py run \
  --manifest .../run_manifest.jsonl --scene_list tools/metrics/sana_wm/smoke_scenes.txt \
  --method_name my_baseline --split simple_60s
python tools/metrics/sana_wm/speed_bench.py run ... --method_name my_variant \
  --step 40 --extra_args='--cache_camera_geometry --cfg_truncate_ratio 0.5'

# 3. Quality evals (separate env for VBench — see notes below), then compare
python tools/metrics/sana_wm/speed_bench.py compare \
  --baseline_dir results/my_baseline/simple_60s \
  --variant_dir results/my_variant/simple_60s --tier0 exact
```

Quality gates (matched scene subsets only — never compare VBench/pose means
across different scene sets): VBench Overall Δ ≥ −0.3 · RotErr/TransErr/
CamMC ≤ +5% relative · revisit LPIPS Δ ≤ +0.005 · ΔIQ not worse by >10%.
Tier-0 (`--tier0 exact|tolerance`) checks stage-1 latent equivalence for
lossless changes and doubles as a pipeline-determinism check.

`profile_stage1.py` hook-times every block (GDN vs softmax vs FFN) over real
denoising steps at benchmark shape.

## Environment notes (hard-won)

- VBench and the model need **separate conda envs** (vbench downgrades
  numpy/transformers under the repo pins); eval env also needs
  `imageio-ffmpeg` and `setuptools==69.5.1` (`pkg_resources.packaging`).
- flash-attn may need `FLASH_ATTENTION_FORCE_BUILD=TRUE` (prebuilt-wheel 404s).
- TransformerEngine pin for torch 2.9.1: `transformer_engine[pytorch]==2.8`.
- `--compile` + NVFP4/fp8 conversion requires
  `SANA_WM_TE_NVFP4_NORMALIZE_MODULE_NAMES=1` (OptimizedModule injects
  `_orig_mod` into module paths) — auto-set by the CLI.

## What's next (see SPEED_ROADMAP.md for full detail)

Biggest untouched levers, in ceiling order: few-step **distillation** of the
bidirectional model (~2.5–3.5× more; the gates here are its acceptance
tests) · **fused GDN kernel** (58% of step time in both models; Triton
prototype → CuTe DSL) · **CUDA-graphing the streaming AR step** + host
overhead recovery · **Blackwell fp4** (path already in-repo) · multi-GPU
latency splitting.
