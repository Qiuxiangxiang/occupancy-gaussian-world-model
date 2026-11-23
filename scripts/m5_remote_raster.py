"""M5 remote raster benchmark — runs ON the GPU host (ea-4090-ts).

Renders a Cross3DGS PLY with gsplat at fixed resolution, warmup + repeats,
CUDA-event timing, and explicit GPU memory accounting. Reads LOD level
subsets from the same PLY (deterministic stride decimation) so every level
has identical inputs except gaussian count.

Memory definitions (recorded, per PLAN M5):
- vram_allocated_mb: torch.cuda.max_memory_allocated around the render loop
- vram_reserved_mb: torch.cuda.max_memory_reserved
- process_rss_mb_delta: /proc/self/status VmHWM delta is NOT used; RSS is
  process-level and noisy. GPU numbers are the primary definition.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_ply(path: Path):
    """Minimal Cross3DGS PLY reader (59 float props, matches repo contract)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler.cross3dgs_ply import read_cross3dgs_ply
    return read_cross3dgs_ply(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--levels", type=int, nargs="+", required=True,
                        help="gaussian counts to render (LOD levels)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from gsplat import rasterization

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda unavailable"}), file=sys.stderr)
        return 2
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)

    cloud = load_ply(args.ply)
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    n_total = len(xyz)
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    quats = torch.tensor(np.asarray(cloud.rot_wxyz), dtype=torch.float32, device=device)
    # Cross3DGS stores scales as natural logs; activate for the rasterizer.
    scales = torch.exp(torch.tensor(np.asarray(cloud.log_scale), dtype=torch.float32,
                                    device=device))
    opac_logits = torch.tensor(np.asarray(cloud.opacity_logit),
                               dtype=torch.float32, device=device)
    # Colors: the exchange file stores SH DC coefficients; the rasterizer
    # takes RGB directly (rgb = 0.5 + SH_C0 * f_dc, the repo's inverse map).
    SH_C0 = 0.28209479177387814
    colors = torch.tensor(0.5 + SH_C0 * np.asarray(cloud.f_dc),
                          dtype=torch.float32, device=device)

    focal = float(max(args.width, args.height))
    K = torch.tensor([[focal, 0, args.width / 2.0],
                      [0, focal, args.height / 2.0],
                      [0, 0, 1]], dtype=torch.float32, device=device)
    viewmat = torch.eye(4, dtype=torch.float32, device=device)
    extent = float(np.abs(xyz).max()) + 1e-6
    viewmat[2, 3] = float(np.linalg.norm(xyz.mean(axis=0)) + 2.0 * extent)

    results = []
    for level in args.levels:
        count = int(level)
        if count > n_total:
            print(json.dumps({"error": f"level {count} > cloud size {n_total}"}),
                  file=sys.stderr)
            return 2
        stride = max(1, n_total // count)
        sel = torch.arange(0, n_total, stride, device=device)[:count]
        m = means[sel].contiguous()
        q = quats[sel].contiguous()
        s = scales[sel].contiguous()
        o = torch.sigmoid(opac_logits[sel]).contiguous()

        def render():
            return rasterization(m, q, s, o, colors[sel].contiguous(),
                                 viewmat[None], K[None],
                                 width=args.width, height=args.height)[0]

        for _ in range(args.warmup):
            render()
        torch.cuda.synchronize()
        start = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        stop = [torch.cuda.Event(enable_timing=True) for _ in range(args.repeats)]
        torch.cuda.reset_peak_memory_stats()
        base_alloc = torch.cuda.memory_allocated()
        for rep in range(args.repeats):
            start[rep].record()
            render()
            stop[rep].record()
        torch.cuda.synchronize()
        lat_ms = sorted(start[r].elapsed_time(stop[r]) for r in range(args.repeats))
        results.append({
            "gaussians": count,
            "resolution": [args.width, args.height],
            "latency_ms_median": float(lat_ms[len(lat_ms) // 2]),
            "latency_ms_p95": float(lat_ms[int(len(lat_ms) * 0.95)]),
            "latency_ms_min": float(lat_ms[0]),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "vram_allocated_mb": float(torch.cuda.max_memory_allocated() - base_alloc) / 1e6,
            "vram_reserved_mb": float(torch.cuda.max_memory_reserved()) / 1e6,
        })
        del m, q, s, o
        torch.cuda.empty_cache()

    report = {
        "gpu": torch.cuda.get_device_name(0),
        "rasterizer": "gsplat",
        "backend": "3dgs",
        "seed": args.seed,
        "source_ply_total_gaussians": int(n_total),
        "memory_definition": "torch.cuda.max_memory_allocated around render loop; reserved via memory_reserved",
        "levels": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
