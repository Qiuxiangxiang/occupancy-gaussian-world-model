"""M5 telemetry driver: physics cost here, raster cost on the 4090 host.

Two sides, one report:

- Physics (this machine, CPU): per-step integrator cost and per-assemble
  modal cost vs gaussian count, RSS memory. This is the occupancy side that
  exists today; the repo still has no occupancy rasterizer and M5 does not
  add one.
- Raster (ea-4090-ts, RTX 4090): gsplat 3DGS rasterization of the exported
  Cross3DGS PLY at LOD levels, CUDA-event latency and VRAM, fixed
  resolution/warmup/repeats/seed (PLAN M5 requirements).

Usage: python -m compiler.m5_telemetry   (skips the remote half without ssh)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import config as C
from compiler.cross3dgs_ply import cloud_from_activated, write_cross3dgs_ply

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "m5_report.json"

RASTER_TOTAL = 100_000
RASTER_LEVELS = [100_000, 50_000, 25_000, 10_000, 2_500]
RASTER_WIDTH, RASTER_HEIGHT = 800, 600
WARMUP, REPEATS = 20, 100
PHYSICS_COUNTS = [64, 128, 256]


def generate_cloud(path: Path, n: int, *, seed: int = 0) -> None:
    """Uniform random cloud in a unit cube: the LOD input, nothing physical."""
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-0.5, 0.5, size=(n, 3))
    scale = np.full((n, 3), 0.01)                  # linear; compiler stores log
    alpha = np.full(n, 0.5)                        # linear; compiler stores logit
    quat = np.zeros((n, 4))
    quat[:, 0] = 1.0
    cloud = cloud_from_activated(xyz, scale, alpha, quat,
                                 np.zeros((n, 3)))
    write_cross3dgs_ply(path, cloud)


def benchmark_physics(counts: list[int]) -> list[dict]:
    """Modal assembly cost vs N. tracemalloc peak is the memory definition;
    ru_maxrss is a process high-water mark and overstates per-scene growth."""
    import tracemalloc
    from occupancy_gaussian_framework import (
        OccupancyGaussian,
        assemble_K_M,
        langevin_step,
        State,
    )
    rows = []
    for n in counts:
        rng = np.random.default_rng(0)
        pos = rng.uniform(-0.5, 0.5, size=(n, 3))
        gaussians = [OccupancyGaussian(pos=pos[i], scale=np.ones(3) * 0.01,
                                       mass=1.0) for i in range(n)]
        state = State(pos=pos, vel=np.zeros((n, 3)))
        masses = np.ones(n)
        tracemalloc.start()
        t0 = time.perf_counter()
        assemble_K_M(gaussians)
        assemble_s = time.perf_counter() - t0
        _, peak_mb = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        steps = 20
        t0 = time.perf_counter()
        for _ in range(steps):
            zero = np.zeros((n, 3))
            state = langevin_step(state, zero, C.DT, masses, T=0.0, gamma=0.0)
        step_s = (time.perf_counter() - t0) / steps
        rows.append({
            "gaussians": n,
            "assemble_K_M_s": assemble_s,
            "integrator_step_s": step_s,
            "assembly_peak_alloc_mb": peak_mb / 1e6,
            "machine": "local-cpu",
            "definition": "wall time single thread; memory via tracemalloc peak",
        })
    return rows


REMOTE_PYTHON = "/data/Users/AO/eacomp/liuqinghua/m5/venv/bin/python"


def run_remote_raster(ply: Path, levels: list[int], host: str) -> dict | None:
    # liuqinghua's convention on this host: his content lives under
    # /data/Users/AO/eacomp/liuqinghua/ (see AGENTS.md there).
    remote_host, _, remote_path = host.partition(":")
    remote_path = remote_path or "/data/Users/AO/eacomp/liuqinghua/m5/repo"
    sync = subprocess.run(
        ["rsync", "-az", "--exclude", ".git", "--exclude", "__pycache__",
         "--exclude", ".pytest_cache", "-e", "ssh -o BatchMode=yes",
         str(ROOT) + "/", f"{remote_host}:{remote_path}/"],
        capture_output=True, text=True, timeout=300)
    if sync.returncode != 0:
        return {"skipped": f"rsync failed: {sync.stderr.strip()[:200]}"}
    remote_ply = f"{remote_path}/{ply.relative_to(ROOT)}"
    out = f"{remote_path}/artifacts/m5_raster_report.json"
    cmd = (f"cd {remote_path} && {REMOTE_PYTHON} "
           f"scripts/m5_remote_raster.py --ply {remote_ply} --output {out} "
           f"--width {RASTER_WIDTH} --height {RASTER_HEIGHT} "
           f"--levels {' '.join(str(v) for v in levels)} "
           f"--warmup {WARMUP} --repeats {REPEATS}")
    run = subprocess.run(["ssh", "-o", "BatchMode=yes", remote_host, cmd],
                         capture_output=True, text=True, timeout=1800)
    if run.returncode != 0:
        return {"skipped": f"remote run failed: {run.stderr.strip()[:300]}"}
    fetch = subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes",
         f"{remote_host}:{out}", str(ARTIFACT.parent / "m5_raster_report.json")],
        capture_output=True, text=True, timeout=120)
    if fetch.returncode != 0:
        return {"skipped": f"report fetch failed: {fetch.stderr.strip()[:200]}"}
    return json.loads((ARTIFACT.parent / "m5_raster_report.json").read_text())


def run_m5(host: str | None = "ea-4090-ts", artifact: Path = ARTIFACT) -> dict:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        ply = Path(tmp) / "lod_input.ply"
        generate_cloud(ply, RASTER_TOTAL, seed=0)
        local_ply = ARTIFACT.parent / "m5_lod_input.ply"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        local_ply.write_bytes(ply.read_bytes())
        physics = benchmark_physics(PHYSICS_COUNTS)
        raster = (run_remote_raster(local_ply, RASTER_LEVELS, host)
                  if host else {"skipped": "no remote host configured"})
    report = {
        "milestone": "M5",
        "resolution": [RASTER_WIDTH, RASTER_HEIGHT],
        "warmup": WARMUP,
        "repeats": REPEATS,
        "seed": 0,
        "same_inputs_across_levels": "stride decimation of one exported PLY; only gaussian count varies",
        "physics_cpu": physics,
        "raster_gpu": raster,
        "open": (
            "No occupancy splat rasterizer exists in this repository; the "
            "3DGS side is measured with gsplat. An occupancy-versus-3DGS "
            "frame comparison stays open until an occupancy renderer exists."
        ),
    }
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[M5] physics levels={len(physics)} raster={'ok' if 'levels' in raster else raster} -> {artifact}")
    return report


if __name__ == "__main__":
    host = sys.argv[1] if len(sys.argv) > 1 else "ea-4090-ts"
    report = run_m5(host or None)
    raise SystemExit(0)
