# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-module CUDA-time profile of one SANA-WM stage-1 denoising step.

Attaches CUDA-event hooks to every DiT block's self-attention, cross-attention,
and FFN, runs a few real denoising steps on the demo scene at benchmark
resolution, and reports where the step time goes — split into softmax-attention
blocks (every ``softmax_every_n``-th) vs GDN blocks vs cross-attn vs FFN.
The first step is discarded (Triton JIT warmup).

Usage (from the repo root, sana env):
  python tools/metrics/sana_wm/profile_stage1.py --num_frames 961 --steps 4 \
    --out results/phase1_profile.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

# Must be set before any sana/diffusion import (mirrors inference_sana_wm.py):
# the xformers cross-attention path asserts on tensor y_lens.
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "inference_video_scripts" / "wm"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, default=961)
    p.add_argument("--steps", type=int, default=4, help="Denoising steps; step 0 is discarded as warmup.")
    p.add_argument("--cfg_scale", type=float, default=5.0)
    p.add_argument("--image", type=Path, default=REPO_ROOT / "asset/sana_wm/demo_0.png")
    p.add_argument("--prompt", type=Path, default=REPO_ROOT / "asset/sana_wm/demo_0.txt")
    p.add_argument("--camera", type=Path, default=REPO_ROOT / "asset/sana_wm/demo_0_pose.npy")
    p.add_argument("--intrinsics", type=Path, default=REPO_ROOT / "asset/sana_wm/demo_0_intrinsics.npy")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    import pyrallis
    from PIL import Image

    import diffusion.model.nets  # noqa: F401  (registers models)
    from inference_sana_wm import (
        HF_DEFAULTS,
        GenerationParams,
        InferenceConfig,
        SanaWMPipeline,
        load_intrinsics,
        resize_and_center_crop,
    )
    from sana.tools.hf_utils import resolve_hf_path

    device = torch.device("cuda")
    config: InferenceConfig = pyrallis.parse(
        config_class=InferenceConfig, config_path=resolve_hf_path(HF_DEFAULTS["config"]), args=[]
    )
    pipeline = SanaWMPipeline(
        config=config,
        model_path=resolve_hf_path(HF_DEFAULTS["model_path"]),
        device=device,
        refiner=None,  # stage-1 only
    )
    model = pipeline.model
    softmax_every_n = getattr(config.model, "softmax_every_n", 4)

    # ---- hook instrumentation: (category, call_idx) -> list of event pairs ----
    records: dict[str, list[tuple[int, torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
    call_counter = {"n": -1}  # incremented on each model forward

    def model_pre_hook(_m, _inp):
        call_counter["n"] += 1

    def make_hooks(category: str):
        pending: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

        def pre(_m, _inp):
            ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            ev[0].record()
            pending.append(ev)

        def post(_m, _inp, _out):
            start, end = pending.pop()
            end.record()
            records[category].append((call_counter["n"], start, end))

        return pre, post

    model.register_forward_pre_hook(model_pre_hook)
    for i, blk in enumerate(model.blocks):
        is_softmax = softmax_every_n > 0 and (i + 1) % softmax_every_n == 0
        for name, module in (
            ("attn_softmax" if is_softmax else "attn_gdn", blk.attn),
            ("cross_attn", blk.cross_attn),
            ("ffn", blk.mlp),
        ):
            if module is None:
                continue
            pre, post = make_hooks(name)
            module.register_forward_pre_hook(pre)
            module.register_forward_hook(post)

    # step totals via the model itself
    pre, post = make_hooks("model_total")
    model.register_forward_pre_hook(pre)
    model.register_forward_hook(post)

    # ---- one real stage-1 run at benchmark shape ----
    image = Image.open(args.image).convert("RGB")
    cropped, src_size, resized_size, crop_offset = resize_and_center_crop(image)
    c2w = np.load(args.camera).astype(np.float32)[: args.num_frames]
    from inference_sana_wm import transform_intrinsics_for_crop

    intr = load_intrinsics(args.intrinsics, c2w.shape[0])
    intr = transform_intrinsics_for_crop(intr, src_size, resized_size, crop_offset)
    prompt = args.prompt.read_text(encoding="utf-8").strip()

    params = GenerationParams(
        num_frames=c2w.shape[0], step=args.steps, cfg_scale=args.cfg_scale, sampling_algo="flow_euler_ltx"
    )
    pipeline.generate(cropped, prompt, c2w, intr, params)
    torch.cuda.synchronize()

    # ---- aggregate, discarding the warmup forward(s) of step 0 ----
    # With CFG batched, one forward == one step; call 0 is warmup.
    sums: dict[str, float] = defaultdict(float)
    counted_calls = set()
    for category, entries in records.items():
        for call_idx, start, end in entries:
            if call_idx == 0:
                continue
            counted_calls.add(call_idx)
            sums[category] += start.elapsed_time(end)
    n_calls = max(len(counted_calls), 1)

    total = sums.get("model_total", 0.0)
    lines = [
        f"stage-1 per-step profile ({args.num_frames} frames, cfg {args.cfg_scale}, "
        f"{n_calls} steps counted after warmup)",
        f"{'category':14s} {'ms/step':>10s} {'share':>7s}",
    ]
    shown = 0.0
    for cat in ("attn_gdn", "attn_softmax", "cross_attn", "ffn"):
        ms = sums.get(cat, 0.0) / n_calls
        shown += ms
        share = 100 * sums.get(cat, 0.0) / total if total else 0.0
        lines.append(f"{cat:14s} {ms:10.1f} {share:6.1f}%")
    other = total / n_calls - shown
    lines.append(f"{'other':14s} {other:10.1f} {100 * other * n_calls / total if total else 0:6.1f}%")
    lines.append(f"{'model_total':14s} {total / n_calls:10.1f} {100.0:6.1f}%")
    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
