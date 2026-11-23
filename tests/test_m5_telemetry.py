"""M5 telemetry: local physics benchmark, LOD cloud export, report schema.

The raster half runs on the 4090 host and is exercised by
`python -m compiler.m5_telemetry ea-4090-ts`; these tests cover everything
that runs on this machine plus the recorded measurement definitions.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from compiler.cross3dgs_ply import read_cross3dgs_ply
from compiler.m5_telemetry import (
    PHYSICS_COUNTS,
    RASTER_LEVELS,
    RASTER_TOTAL,
    benchmark_physics,
    generate_cloud,
    run_m5,
)


def test_lod_cloud_roundtrip_contract(tmp_path):
    ply = tmp_path / "lod.ply"
    generate_cloud(ply, 5000, seed=0)
    cloud = read_cross3dgs_ply(ply)
    assert cloud.xyz.shape == (5000, 3)
    # Cross3DGS contract: scales stored as logs, opacity as logit
    assert np.allclose(np.exp(cloud.log_scale), 0.01, atol=1e-6)
    assert np.allclose(1.0 / (1.0 + np.exp(-cloud.opacity_logit)), 0.5, atol=1e-6)
    assert np.allclose(np.abs(cloud.rot_wxyz[:, 0]), 1.0)  # identity quats


def test_lod_levels_are_subsets_of_one_cloud():
    assert RASTER_LEVELS[0] == RASTER_TOTAL
    assert all(level <= RASTER_TOTAL for level in RASTER_LEVELS)
    assert RASTER_LEVELS == sorted(RASTER_LEVELS, reverse=True)


def test_physics_benchmark_scales_and_reports_memory():
    rows = benchmark_physics(PHYSICS_COUNTS)
    assert [r["gaussians"] for r in rows] == PHYSICS_COUNTS
    # O(N^2) assembly: 4x N should cost clearly more than 2x
    ratio = rows[-1]["assemble_K_M_s"] / rows[-2]["assemble_K_M_s"]
    assert ratio > 2.0, f"assembly did not show superlinear growth: {ratio}"
    for row in rows:
        assert row["assembly_peak_alloc_mb"] > 0.0
        assert "tracemalloc" in row["definition"]
        assert row["integrator_step_s"] > 0.0


def test_run_m5_without_remote_marks_raster_skipped(tmp_path):
    report = run_m5(None, artifact=tmp_path / "m5_report.json")
    assert (tmp_path / "m5_report.json").is_file()
    assert report["physics_cpu"], "physics side must run locally"
    assert "skipped" in report["raster_gpu"]
    assert "No occupancy splat rasterizer" in report["open"]
    # Fixed methodology fields the PLAN requires
    assert report["resolution"] == [800, 600]
    assert report["warmup"] > 0 and report["repeats"] > 0
    assert report["seed"] == 0
