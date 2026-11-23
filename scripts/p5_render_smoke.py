"""G5c render-side LOD smoke — runs ON the GPU host.

Stride-decimates one exported cloud at each LOD level, renders a single
frame per level with gsplat, and reports pixel coverage (fraction of
non-background pixels). Rendering-side only: no dynamics claim, no
equivalence claim. Coverage is reported per level; monotonicity is
recorded, not gated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--levels", type=int, nargs="+", required=True)
    args = parser.parse_args()

    import torch
    from gsplat import rasterization

    if not torch.cuda.is_available():
        print(json.dumps({"error": "cuda unavailable"}), file=sys.stderr)
        return 2
    device = torch.device("cuda:0")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from compiler.cross3dgs_ply import read_cross3dgs_ply
    cloud = read_cross3dgs_ply(args.ply)
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    n_total = len(xyz)
    means = torch.tensor(xyz, dtype=torch.float32, device=device)
    quats = torch.tensor(np.asarray(cloud.rot_wxyz), dtype=torch.float32, device=device)
    scales = torch.exp(torch.tensor(np.asarray(cloud.log_scale), dtype=torch.float32,
                                    device=device))
    opac_logits = torch.tensor(np.asarray(cloud.opacity_logit),
                               dtype=torch.float32, device=device)
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

    rows = []
    for level in args.levels:
        count = int(level)
        if count > n_total:
            print(json.dumps({"error": f"level {count} > cloud size {n_total}"}),
                  file=sys.stderr)
            return 2
        stride = max(1, n_total // count)
        sel = torch.arange(0, n_total, stride, device=device)[:count]
        render, _alpha, _info = rasterization(
            means[sel].contiguous(), quats[sel].contiguous(),
            scales[sel].contiguous(),
            torch.sigmoid(opac_logits[sel]).contiguous(),
            colors[sel].contiguous(),
            viewmat[None], K[None], width=args.width, height=args.height)
        frame = render[0].detach().cpu().numpy()
        coverage = float(np.mean(np.any(frame > 1e-3, axis=-1)))
        rows.append({"gaussians": count,
                     "coverage_nonblack_fraction": coverage,
                     "frame_value_max": float(frame.max())})
        del render
        torch.cuda.empty_cache()

    report = {"gpu": torch.cuda.get_device_name(0),
              "rasterizer": "gsplat", "backend": "3dgs",
              "resolution": [args.width, args.height],
              "source_ply_total_gaussians": int(n_total),
              "levels": rows,
              "claim": "rendering-side only; no dynamics or equivalence claim"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
