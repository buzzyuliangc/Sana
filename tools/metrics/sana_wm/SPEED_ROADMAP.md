# SANA-WM Bidirectional Inference: Speed Roadmap

Status and plan for latency optimization of the bidirectional pipeline while
holding quality, measured with `speed_bench.py` on pinned SANA-WM-Bench scenes
and gated on VBench (9 dims), revisit consistency, temporal degradation, and
Pi3 camera-pose error. Baselines on 1× H100 80GB.

**Quality gates** (vs baseline, matched scene subsets only):
VBench Overall Δ ≥ −0.3 · RotErr/TransErr/CamMC ≤ +5% relative ·
revisit LPIPS Δ ≤ +0.005 · ΔIQ not worse by >10% relative.

## Wave 1 — DONE (2026-07-08)

Baseline: 920 s / 60 s-scene (961 frames, 60 steps, CFG 5.0); stage-1 = 83%,
refiner = 12%, VAE = 5%.

| Change | Speedup | Gate result |
|---|---|---|
| `--cache_camera_geometry` | 1.013× | bit-exact (Tier-0 PASS) |
| `--cfg_truncate_ratio 0.5` | 1.264× | PASS (TransErr −9%) |
| `--step 40` | 1.380× | borderline alone; PASS in stack |
| **stack B (all three)** | **1.709× (538 s/scene)** | **PASS** (VBench −0.16, RotErr +4.9%) |
| `--compile` | 1.20×/step, 1.04× e2e | parked (per-process warmup; serving-only) |
| step 30 / 20, cfg 0.3 | 1.42–2.23× | REJECTED — pose gates (+22–44% RotErr) |

Key lessons: camera adherence breaks before visual quality (pose gates are the
binding constraint); late-step CFG fights the camera branch; optimize by
profile share, not repeat count. 80-scene × 2-split validation of stack B in
progress.

## Phase 1 — No retraining (target ~1.5–2× more → ~270–360 s/scene)

Post-stack-B profile: stage-1 ~69%, refiner ~21%, VAE+misc ~10%.
Budget ~40–80 H100-hrs. Screening on the 5-scene set, gates as above.

1. **Per-block profile** of a stage-1 step (softmax vs GDN vs FFN share) —
   `profile_stage1.py`. Decides how much items 4–5 and Phase 2 matter.
2. **Flow-DPM-Solver step sweep** (order 2, steps ∈ {20, 25, 30}) vs 40-step
   Euler. Config-only; expected up to 1.3×.
3. **fp8 W8A8** on stage-1 linears via the in-repo TransformerEngine path
   (`SANA_WM_STAGE1_*` env flags; Float8BlockScaling). Try `self_qkv` mode
   first, widen to `self_attn+cross+ffn` if gates pass. Expected 1.15–1.3×.
   Watch item: low-magnitude camera-branch residuals (pose gates detect).
4. **SageAttention** (INT8 exact-algorithm attention, github.com/thu-ml/SageAttention)
   as a drop-in for the 5 softmax blocks. Expected 1.1–1.3×.
5. **Timestep feature caching** (TeaCache, arXiv 2411.19108; FORA; DeepCache):
   reuse deep-block outputs across adjacent mid-trajectory steps. Expected
   1.15–1.35×; partially anticorrelated with step reduction.

## Phase 2 — Light finetuning (~1.3–1.6× more on stage-1)

Requires training stack (configs/sana_wm/stage1/, FSDP2 + CP; Sekai data is
public but budget data-pipeline time). ~400–800 H100-hrs (~$1.5–3k).

- **Windowed softmax + sink frames, finetuned in** — convert the five O(N²)
  softmax blocks (106k tokens) to temporal-window attention with global sink
  frames; brief finetune to adapt. Resources: in-repo WindowAttention /
  block-mask machinery / tools/attn_mask; Sliding-Tile-Attention
  (arXiv 2502.04507) for the video-DiT recipe.
- Alternative at same cost: reduce softmax cadence (softmax_every_n 4→6/8) +
  finetune.

## Phase 3 — Distillation (~2.5–3.5× more → ~60–90 s/scene end-to-end)

4–8-step **bidirectional** student with CFG baked in. Feasibility proven on
this architecture by the released SANA-WM_streaming 4-step student; the
bidirectional student keeps full-sequence attention (global consistency,
revisit fidelity) and removes only step count.

- Methods: sCM (arXiv 2410.11081), LADD, DMD2 (arXiv 2405.14867); in-house
  lineage: SANA-Sprint (arXiv 2503.09641); in-repo streaming distillation code
  paths; Cosmos-RL post-training infra.
- Budget: 32–64 GPUs × days ($10k+), plus the training dataset.
- Risk: camera adherence degradation — 80-scene pose gates are the acceptance
  test.

## Standing alternatives

- **SANA-WM_streaming** (released): ~realtime, ~9× faster than stack B, lower
  temporal-consistency tier (causal + KV window → weaker revisit fidelity).
  Use when interactivity outranks the quality tier.
- **Blackwell hardware**: ~1.5–2× raw + unlocks the in-repo fp4 path.

## Decision points

- After Phase 1 item 1 (profile): softmax share >50% of step → prioritize
  item 4 and Phase 2; GDN-dominated → feature caching and Phase 3 are the
  remaining levers.
- After Phase 1: if composed speedup ≥ ~3× vs original baseline meets the
  product need, stop; Phases 2–3 only pay off for a productized pipeline.

---

# Streaming/causal model (SANA-WM_streaming) roadmap

The streaming pipeline is an overlapped 3-stream assembly line (AR Stage-1 ∥
chunk-causal refiner ∥ causal VAE): throughput = slowest stage, not the sum.
Goals split into sustained realtime factor, first-chunk latency, and VRAM.
Already shipped upstream: 4-step distilled student (CFG baked), fp8/fp4
modes, torch.compile, KV-window refiner. Docs: ~0.93-1.09x RT on H100 bf16;
KV window 11→2 gives 1.26x at quality cost; fp4 = Blackwell-only.

**Phase S0 — profile the critical path** (existing --benchmark_json:
per-stage CUDA s, first-chunk latency; check host-side launch gaps).

**Phase S1 — no retraining**: CUDA graphs on the fixed-shape AR step
(10-25% on small-chunk loops); balance the slowest stage only (refiner: KV
window sweep 11→8→5 / fp8-refiner / SageAttention; stage1: widen fp8
coverage); try 3-step --denoising_step_list on the 4-step student; fp8 KV
cache; chunk-size latency/throughput dial; warm-server for first-frame
latency. Gates: standard benchmark + temporal-degradation and revisit
weighted heavier (AR fails by drifting).

**Phase S2 — light training (~$3-8k)**: 2-step/1-step re-distillation
(DMD2-style); train-in KV window 5; depth-pruned student (20→~14 blocks).

**Phase S3 — bigger bets**: Blackwell fp4 (already implemented — realtime
on consumer 32GB); multi-GPU pipeline parallelism (one stage per GPU);
larger-patch student (shared with bidirectional Phase 3).

---

# Streaming kernel plan (detail for Phase S1 items)

Regime: ~2,640 tokens/chunk (3 latent frames), 4 steps x 20 blocks per chunk
— launch-overhead + memory-bound; opposite of the bidirectional regime.
All gains are hypotheses until S0 measures them.

- **S0 profile** (2 days): torch.profiler + nsys per chunk — launch-gap share,
  GDN/softmax/GEMM/elementwise split, refiner host-side KV-cache cost,
  inter-stream bubbles. Decides K1-vs-K2 priority.
- **K4 CUDA graphs** (1-2 wks, 1.1-1.25x pipeline): graph-capture the static
  per-chunk step; prerequisite = fixed-size KV/state buffers (shared with K2).
- **K1 fused streaming-GDN kernel** (4-8 wks, 1.3-1.8x stage-1): fuse
  conv->gates->delta update->read->gate for the cached chunk-causal GDN
  (state 20 heads x 112x112, tensor-core aligned). Stage order: reference
  oracle tests -> Triton fusion prototype (off-ramp if >80% of gain) ->
  CuTe DSL port (TMA + WGMMA warp-specialized, state SMEM-resident) ->
  fp8-IO variant. Oracle = existing Triton kernels; same Tier-0 discipline.
- **K2 ring-buffer window attention + fp8 KV** (2-4 wks, 1.3-1.7x refiner):
  cheap half first (pre-allocated ring KV + copy_, no kernel — enables K4);
  then CuTe/FA3-style kernel with in-kernel wraparound, pinned sink rows,
  fp8 KV with per-block scales.
- **K3 elementwise/epilogue fusions** (1-3 wks, 1.05-1.15x): adaLN modulate,
  QK norms, GLUMBConvTemp chain; do last (earlier work changes the op mix).

Composed: ~1.5-2.5x realtime headroom on H100, no weight changes.
Resourcing: S0+K4+K2-cheap = ~3-4 wks generalist; K1-CuTe + K2-kernel =
1-2 months specialist, gated on the Triton prototype's result.

## S0 results (2026-07-10, H100 pod, 321-frame demo, discard I/O)

- Baseline: 0.66x steady-state RT writing to disk; **0.79x with output discarded**
  → MP4 encoding sits on the critical path (~15%); local disk ≈ network volume,
  so it is encode cost, not storage. S1 fix: async writer thread.
- Per-stage CUDA s (serialized profile, 313 frames): **stage1 34.6 (58%) ·
  refiner 17.1 (29%) · decode 8.1 (13%)**. Stage-1 GPU work alone (34.6s)
  exceeds the clip's realtime budget (19.5s) → no pipelining fix can reach 1x
  RT; stage-1 must get cheaper. Kernel priority confirmed: K4 → K1 →
  stage1-fp8; K2 second tier.
- Published 1.09x vs our 0.79x: partially unexplained; confirm with a
  961-frame run (better warmup amortization) before attributing to host
  overhead.
- 961-frame confirmation (benchmark scene, discard): steady 0.683x — the gap
  vs published 1.09x is REAL on this host, not warmup amortization →
  raises K4's expected value; add a config/host diff vs upstream measurement
  (bare-metal vs virtualized CPU, clock pinning) to the S1 checklist.
  First-chunk hit 42.6s on the new shape: set TORCHINDUCTOR_CACHE_DIR on
  pod 2 to persist compiles across processes/shapes.
