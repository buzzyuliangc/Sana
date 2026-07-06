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

"""SANA-WM speed benchmark: smoke-set runner + baseline A/B comparison.

Runs a pinned list of benchmark scenes through the bidirectional inference
script (one subprocess per scene for clean VRAM peaks), collecting per-scene
``--timing_json`` and ``--save_stage1_latents`` artifacts, then compares a
variant run against a baseline run: per-stage latency, median paired speedup,
peak VRAM, Tier-0 stage-1 latent equivalence, and (when the quality
evaluators have been run) VBench / pose / revisit deltas.

Subcommands:
  pick-smoke  Pin a smoke-scene list (N per category) from a bench manifest.
  run         Generate scenes for one variant config.
  compare     Compare a variant run directory against a baseline run directory.

Typical flow (from the repo root):
  python tools/metrics/sana_wm/speed_bench.py pick-smoke \
    --manifest data/SANA-WM-Bench/benchmark_v2_smooth_60s/sanawm_export_v2/run_manifest.jsonl \
    --per_category 2 --out tools/metrics/sana_wm/smoke_scenes.txt

  python tools/metrics/sana_wm/speed_bench.py run \
    --manifest data/SANA-WM-Bench/benchmark_v2_smooth_60s/sanawm_export_v2/run_manifest.jsonl \
    --scene_list tools/metrics/sana_wm/smoke_scenes.txt \
    --method_name sana_wm_baseline --split simple_60s

  python tools/metrics/sana_wm/speed_bench.py compare \
    --baseline_dir results/sana_wm_baseline/simple_60s \
    --variant_dir results/sana_wm_cachegeom/simple_60s \
    --tier0 exact

Result layout matches the existing evaluators (eval_unified.py):
  results/<method>/<split>/<scene>_generated.mp4
  results/<method>/<split>/speed/<scene>_timing.json
  results/<method>/<split>/speed/<scene>_stage1_latent.pt
  results/<variant>/<split>/speed_bench.{json,md}
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SCRIPT = REPO_ROOT / "inference_video_scripts" / "wm" / "inference_sana_wm.py"

# Official 80-scene benchmark generation settings (docs/sana-wm-bench.md).
BENCH_DEFAULTS = {
    "num_frames": 961,
    "fps": 16,
    "step": 60,
    "cfg_scale": 5.0,
    "flow_shift": 8.0,
    "sampling_algo": "flow_euler_ltx",
    "seed": 42,
    "refiner_seed": 42,
}


# ---------------------------------------------------------------------------
# Manifest handling
# ---------------------------------------------------------------------------


def load_manifest(manifest: Path) -> list[dict]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"Manifest is empty: {manifest}")
    return rows


def scene_category(scene_id: str) -> str:
    """``game_style_001`` -> ``game_style`` (public IDs are ``<category>_NNN``)."""
    m = re.match(r"^(.*)_(\d+)$", scene_id)
    return m.group(1) if m else scene_id


def bench_root_for(manifest: Path) -> Path:
    # <bench_root>/<split_dir>/sanawm_export_v2/run_manifest.jsonl
    return manifest.resolve().parents[2]


def resolve_row_path(row: dict, keys: tuple[str, ...], bench_root: Path, what: str) -> Path:
    for key in keys:
        value = row.get(key)
        if value:
            path = Path(value)
            if not path.is_absolute():
                path = bench_root / path
            if path.exists():
                return path
            raise SystemExit(f"{what} for scene {row.get('id')!r} not found: {path}")
    raise SystemExit(
        f"Manifest row for scene {row.get('id')!r} has no {what} key "
        f"(tried {keys}); available keys: {sorted(row.keys())}"
    )


def read_scene_list(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            scenes.append(line)
    if not scenes:
        raise SystemExit(f"No scenes in {path} — run the pick-smoke subcommand first.")
    return scenes


# ---------------------------------------------------------------------------
# pick-smoke
# ---------------------------------------------------------------------------


def cmd_pick_smoke(args: argparse.Namespace) -> None:
    rows = load_manifest(args.manifest)
    by_category: dict[str, list[str]] = {}
    for row in rows:
        by_category.setdefault(scene_category(row["id"]), []).append(row["id"])
    picked: list[str] = []
    for category in sorted(by_category):
        picked.extend(sorted(by_category[category])[: args.per_category])
    lines = [
        "# SANA-WM speed-bench smoke set: pinned scene IDs, one per line.",
        f"# Generated by speed_bench.py pick-smoke --per_category {args.per_category}",
        f"# from {args.manifest}",
        *picked,
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Pinned {len(picked)} scenes ({len(by_category)} categories) -> {args.out}")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def prepare_scene_inputs(row: dict, bench_root: Path, inputs_dir: Path) -> dict[str, Path]:
    """Materialize the file-based inputs the inference CLI requires."""
    import numpy as np

    scene = row["id"]
    inputs_dir.mkdir(parents=True, exist_ok=True)

    image = resolve_row_path(row, ("image_path", "image"), bench_root, "image")
    camera_npz = resolve_row_path(row, ("camera_path", "camera"), bench_root, "camera")

    prompt_path = inputs_dir / f"{scene}.txt"
    prompt_path.write_text(str(row["prompt"]), encoding="utf-8")

    traj = np.load(camera_npz)
    c2w_path = inputs_dir / f"{scene}_c2w.npy"
    intr_path = inputs_dir / f"{scene}_intrinsics.npy"
    np.save(c2w_path, traj["c2w"])
    np.save(intr_path, traj["intrinsics"])

    return {"image": image, "prompt": prompt_path, "camera": c2w_path, "intrinsics": intr_path}


def run_scene(
    scene: str,
    inputs: dict[str, Path],
    out_dir: Path,
    gen_args: dict[str, object],
    extra_args: list[str],
    *,
    warmup: bool,
) -> dict[str, object]:
    speed_dir = out_dir / "speed"
    speed_dir.mkdir(parents=True, exist_ok=True)
    timing_json = speed_dir / f"{scene}_timing.json"
    latents_pt = speed_dir / f"{scene}_stage1_latent.pt"

    cmd = [
        sys.executable,
        str(INFER_SCRIPT),
        "--image", str(inputs["image"]),
        "--prompt", str(inputs["prompt"]),
        "--camera", str(inputs["camera"]),
        "--intrinsics", str(inputs["intrinsics"]),
        "--output_dir", str(out_dir),
        "--name", scene,
        "--no_action_overlay",
        "--timing_json", str(timing_json),
        "--save_stage1_latents", str(latents_pt),
    ]
    for key, value in gen_args.items():
        cmd += [f"--{key}", str(value)]
    cmd += extra_args

    print(f"[speed_bench] {scene}: {' '.join(shlex.quote(c) for c in cmd[2:])}", flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    ok = proc.returncode == 0 and timing_json.exists()
    if ok and warmup:
        payload = json.loads(timing_json.read_text(encoding="utf-8"))
        payload["warmup"] = True
        timing_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"scene": scene, "ok": ok, "returncode": proc.returncode, "warmup": warmup}


def parse_sweep(spec: str) -> list[dict[str, str]]:
    """``"step=20,30;flow_shift=8,9.8"`` -> cartesian product of overrides."""
    axes: list[list[tuple[str, str]]] = []
    for axis in spec.split(";"):
        axis = axis.strip()
        if not axis:
            continue
        key, _, values = axis.partition("=")
        axes.append([(key.strip(), v.strip()) for v in values.split(",") if v.strip()])
    return [dict(combo) for combo in itertools.product(*axes)] if axes else [{}]


def cmd_run(args: argparse.Namespace) -> None:
    rows = {row["id"]: row for row in load_manifest(args.manifest)}
    bench_root = bench_root_for(args.manifest)
    scenes = args.scenes.split(",") if args.scenes else read_scene_list(args.scene_list)
    missing = [s for s in scenes if s not in rows]
    if missing:
        raise SystemExit(f"Scenes not in manifest: {missing}")

    gen_args = dict(BENCH_DEFAULTS)
    for key in ("num_frames", "fps", "step", "cfg_scale", "flow_shift", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            gen_args[key] = value
    gen_args["sampling_algo"] = args.sampling_algo
    extra_args = shlex.split(args.extra_args) if args.extra_args else []

    for overrides in parse_sweep(args.sweep) if args.sweep else [{}]:
        method = args.method_name + "".join(f"_{k}{v}" for k, v in sorted(overrides.items()))
        out_dir = args.results_root / method / args.split
        out_dir.mkdir(parents=True, exist_ok=True)
        scene_args = {**gen_args, **overrides}
        session: list[dict[str, object]] = []
        for idx, scene in enumerate(scenes):
            inputs = prepare_scene_inputs(rows[scene], bench_root, out_dir / "speed" / "_inputs")
            warmup = args.mark_first_warmup and idx == 0
            session.append(run_scene(scene, inputs, out_dir, scene_args, extra_args, warmup=warmup))
        (out_dir / "speed" / "session.json").write_text(
            json.dumps({"method": method, "gen_args": scene_args, "extra_args": extra_args, "runs": session}, indent=2),
            encoding="utf-8",
        )
        failed = [r["scene"] for r in session if not r["ok"]]
        print(f"[speed_bench] {method}: {len(session) - len(failed)}/{len(session)} scenes OK"
              + (f", FAILED: {failed}" if failed else ""))
        if failed:
            sys.exit(1)


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def load_timings(split_dir: Path) -> dict[str, dict]:
    out = {}
    for path in sorted((split_dir / "speed").glob("*_timing.json")):
        scene = path.name[: -len("_timing.json")]
        out[scene] = json.loads(path.read_text(encoding="utf-8"))
    if not out:
        raise SystemExit(f"No timing JSONs under {split_dir}/speed — run the run subcommand first.")
    return out


def stage_seconds(timing: dict, stage: str) -> float | None:
    record = timing.get("stages", {}).get(stage)
    return None if record is None else record.get("wall_s")


def step_ms_excl_first(timing: dict) -> float | None:
    steps = timing.get("per_step_ms") or []
    if len(steps) < 2:
        return None
    return sum(steps[1:]) / len(steps[1:])


def tier0_check(baseline_dir: Path, variant_dir: Path, scene: str, mode: str, tol: float) -> dict[str, object]:
    import torch

    base_pt = baseline_dir / "speed" / f"{scene}_stage1_latent.pt"
    var_pt = variant_dir / "speed" / f"{scene}_stage1_latent.pt"
    if not base_pt.exists() or not var_pt.exists():
        return {"status": "missing_latents"}
    a = torch.load(base_pt, map_location="cpu", weights_only=True).float()
    b = torch.load(var_pt, map_location="cpu", weights_only=True).float()
    if a.shape != b.shape:
        return {"status": "FAIL", "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    diff = (a - b).abs()
    max_abs = diff.max().item()
    result = {
        "bitwise_equal": bool(torch.equal(a, b)),
        "max_abs_diff": max_abs,
        "mse": diff.pow(2).mean().item(),
    }
    if mode == "exact":
        result["status"] = "PASS" if result["bitwise_equal"] else "FAIL"
    else:  # tolerance
        result["status"] = "PASS" if max_abs <= tol else "FAIL"
    return result


def load_quality_summary(split_dir: Path, scenes: list[str]) -> dict[str, float]:
    """Best-effort read of eval_unified / pose-eval outputs when present."""
    method_dir, split = split_dir.parent, split_dir.name
    out: dict[str, float] = {}
    vbench = method_dir / "eval" / split / "vbench_scores.json"
    if vbench.exists():
        scores = json.loads(vbench.read_text(encoding="utf-8"))
        if isinstance(scores.get("quality_score"), (int, float)):
            out["vbench_overall"] = scores["quality_score"] * 100.0
    revisit = method_dir / "eval" / split / "revisit_consistency.json"
    if revisit.exists():
        data = json.loads(revisit.read_text(encoding="utf-8"))
        lpips = data.get("mean", {}).get("lpips") if isinstance(data.get("mean"), dict) else data.get("lpips")
        if isinstance(lpips, (int, float)):
            out["revisit_lpips"] = lpips
    poses = split_dir / "eval_poses.json"
    if poses.exists():
        data = json.loads(poses.read_text(encoding="utf-8"))
        per_scene = [v for k, v in data.items() if isinstance(v, dict) and (not scenes or k in scenes)]
        for metric in ("RotErr", "TransErr_rel", "CamMC_rel"):
            values = [v[metric] for v in per_scene if isinstance(v.get(metric), (int, float))]
            if values:
                out[metric] = sum(values) / len(values)
    return out


def _fmt(value: object, spec: str = ".1f") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def cmd_compare(args: argparse.Namespace) -> None:
    base = load_timings(args.baseline_dir)
    var = load_timings(args.variant_dir)
    scenes = sorted(set(base) & set(var))
    if not scenes:
        raise SystemExit("No overlapping scenes between baseline and variant timing JSONs.")
    skipped = sorted((set(base) | set(var)) - set(scenes))
    if skipped:
        print(f"[speed_bench] WARNING: scenes present on one side only, skipped: {skipped}")

    rows, ratios = [], []
    for scene in scenes:
        b, v = base[scene], var[scene]
        warmup = bool(b.get("warmup") or v.get("warmup"))
        b_e2e, v_e2e = b.get("end_to_end_s"), v.get("end_to_end_s")
        speedup = (b_e2e / v_e2e) if b_e2e and v_e2e else None
        tier0 = (
            tier0_check(args.baseline_dir, args.variant_dir, scene, args.tier0, args.tolerance)
            if args.tier0 != "skip"
            else None
        )
        if speedup is not None and not warmup:
            ratios.append(speedup)
        rows.append(
            {
                "scene": scene,
                "warmup": warmup,
                "baseline_e2e_s": b_e2e,
                "variant_e2e_s": v_e2e,
                "speedup": speedup,
                "baseline_stage1_s": stage_seconds(b, "stage1_sampling"),
                "variant_stage1_s": stage_seconds(v, "stage1_sampling"),
                "baseline_step_ms": step_ms_excl_first(b),
                "variant_step_ms": step_ms_excl_first(v),
                "baseline_refiner_s": stage_seconds(b, "refiner"),
                "variant_refiner_s": stage_seconds(v, "refiner"),
                "baseline_vae_decode_s": stage_seconds(b, "vae_decode"),
                "variant_vae_decode_s": stage_seconds(v, "vae_decode"),
                "baseline_peak_gb": b.get("peak_mem_gb"),
                "variant_peak_gb": v.get("peak_mem_gb"),
                "baseline_model_evals": b.get("model_eval_count"),
                "variant_model_evals": v.get("model_eval_count"),
                "tier0": tier0,
            }
        )

    tier0_statuses = [r["tier0"]["status"] for r in rows if r["tier0"]]
    quality_base = load_quality_summary(args.baseline_dir, scenes)
    quality_var = load_quality_summary(args.variant_dir, scenes)
    quality_delta = {k: quality_var[k] - quality_base[k] for k in quality_base if k in quality_var}

    summary = {
        "baseline_dir": str(args.baseline_dir),
        "variant_dir": str(args.variant_dir),
        "scenes": len(scenes),
        "median_speedup": statistics.median(ratios) if ratios else None,
        "tier0_mode": args.tier0,
        "tier0_pass": (all(s == "PASS" for s in tier0_statuses) if tier0_statuses else None),
        "quality_baseline": quality_base,
        "quality_variant": quality_var,
        "quality_delta": quality_delta,
        "per_scene": rows,
    }
    out_json = args.variant_dir / "speed_bench.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        f"# speed_bench: {args.variant_dir.parent.name} vs {args.baseline_dir.parent.name} ({args.variant_dir.name})",
        "",
        f"- Scenes compared: **{len(scenes)}** (warmup-marked scenes excluded from aggregates)",
        f"- Median paired e2e speedup: **{_fmt(summary['median_speedup'], '.3f')}x**",
        f"- Tier-0 ({args.tier0}): **{_fmt(summary['tier0_pass'])}**",
        "",
        "| scene | e2e s (base→var) | speedup | step ms (base→var) | refiner s | vae s | peak GB | evals | tier0 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        warm = " (warmup)" if r["warmup"] else ""
        tier0_cell = r["tier0"]["status"] if r["tier0"] else "-"
        if r["tier0"] and "max_abs_diff" in r["tier0"]:
            tier0_cell += f" (maxΔ={r['tier0']['max_abs_diff']:.2e})"
        md.append(
            f"| {r['scene']}{warm} "
            f"| {_fmt(r['baseline_e2e_s'])}→{_fmt(r['variant_e2e_s'])} "
            f"| {_fmt(r['speedup'], '.3f')}x "
            f"| {_fmt(r['baseline_step_ms'])}→{_fmt(r['variant_step_ms'])} "
            f"| {_fmt(r['baseline_refiner_s'])}→{_fmt(r['variant_refiner_s'])} "
            f"| {_fmt(r['baseline_vae_decode_s'])}→{_fmt(r['variant_vae_decode_s'])} "
            f"| {_fmt(r['baseline_peak_gb'], '.1f')}→{_fmt(r['variant_peak_gb'], '.1f')} "
            f"| {_fmt(r['baseline_model_evals'], 'd')}→{_fmt(r['variant_model_evals'], 'd')} "
            f"| {tier0_cell} |"
        )
    if quality_delta:
        md += ["", "## Quality deltas (variant − baseline)", ""]
        md += [f"- {k}: {quality_base[k]:.4f} → {quality_var[k]:.4f} (Δ {quality_delta[k]:+.4f})" for k in quality_delta]
    else:
        md += ["", "_No quality eval outputs found; run eval_unified.py + eval_benchmark_poses.py for Tier-1 gates._"]

    out_md = args.variant_dir / "speed_bench.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[speed_bench] wrote {out_md} and {out_json}")
    print("\n".join(md[:12]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(prog="speed_bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    pick = sub.add_parser("pick-smoke", help="Pin a smoke-scene list from a bench manifest.")
    pick.add_argument("--manifest", type=Path, required=True)
    pick.add_argument("--per_category", type=int, default=2)
    pick.add_argument("--out", type=Path, default=Path("tools/metrics/sana_wm/smoke_scenes.txt"))
    pick.set_defaults(func=cmd_pick_smoke)

    run = sub.add_parser("run", help="Generate scenes for one variant config.")
    run.add_argument("--manifest", type=Path, required=True)
    scene_group = run.add_mutually_exclusive_group(required=True)
    scene_group.add_argument("--scenes", type=str, help="Comma-separated scene IDs.")
    scene_group.add_argument("--scene_list", type=Path, help="File with one scene ID per line.")
    run.add_argument("--method_name", required=True)
    run.add_argument("--split", default="simple_60s")
    run.add_argument("--results_root", type=Path, default=Path("results"))
    run.add_argument("--num_frames", type=int, default=None, help=f"Default {BENCH_DEFAULTS['num_frames']}.")
    run.add_argument("--fps", type=int, default=None)
    run.add_argument("--step", type=int, default=None)
    run.add_argument("--cfg_scale", type=float, default=None)
    run.add_argument("--flow_shift", type=float, default=None)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--sampling_algo", default=BENCH_DEFAULTS["sampling_algo"])
    run.add_argument("--extra_args", type=str, default="", help="Extra inference CLI args, shell-quoted.")
    run.add_argument("--sweep", type=str, default=None, help="e.g. 'step=20,30,40,60;flow_shift=8,9.8'")
    run.add_argument("--mark_first_warmup", action=argparse.BooleanOptionalAction, default=True)
    run.set_defaults(func=cmd_run)

    comp = sub.add_parser("compare", help="Compare a variant run against a baseline run.")
    comp.add_argument("--baseline_dir", type=Path, required=True, help="e.g. results/sana_wm_baseline/simple_60s")
    comp.add_argument("--variant_dir", type=Path, required=True)
    comp.add_argument("--tier0", choices=["exact", "tolerance", "skip"], default="skip")
    comp.add_argument("--tolerance", type=float, default=2e-2, help="max|Δ| bound for --tier0 tolerance.")
    comp.set_defaults(func=cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
