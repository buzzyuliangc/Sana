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
