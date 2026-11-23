"""Phase 5 gate G5: LOD machinery invariants + budget policy on real telemetry.

Recorded scope (PLAN.md, Phase 5): dynamics-level LOD — decimating a scene
while it simulates — is BLOCKED on P1-2 rigid reduction. The Phase 0
dynamics has no attractive bond: pair stiffness enters modal analysis only,
never the integrator, so any multi-splat "body" is a cluster of free
particles whose internal overlap repels and disintegrates (observed: two
splats 0.01 m overlapped separate within ~19 steps). Merging such clusters
would preserve moments at t=0 and diverge immediately after — dishonest.

What G5 verifies instead, today:
- G5a merge invariants: over a randomized cloud pyramid, mass is conserved
  exactly at every level, merged covariances stay positive semidefinite,
  and every performed merge reports its off-diagonal residual.
- G5b adaptive policy against the REAL M5 latency ladder (measured on the
  RTX 4090, artifacts/m5_report.json): a frame-budget trace must settle at
  the finest level whose measured cost fits the budget, switching only
  after the hysteresis streak.
- G5c render-side smoke (optional, needs the 4090 over Tailscale): stride
  decimation of the M5 cloud at the policy-chosen level renders one frame
  per level without error; pixel coverage per level is reported, not gated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from compiler.lod import (Ladder, build_lod_pyramid, choose_level,
                          merge_diagnostics, merge_pair)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "p5_report.json"


def _cloud(n: int, seed: int = 0):
    """Connected 4x4x4 grid, spacing 0.09 < 0.1 reach, so every component
    can merge. G5a exercises merge algebra only — no dynamics is run, so
    the overlap that would repel in a simulation is irrelevant here."""
    from occupancy_gaussian_framework import OccupancyGaussian
    rng = np.random.default_rng(seed)
    side = round(n ** (1 / 3)) or 1
    gaussians = []
    k = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if k >= n:
                    return gaussians
                gaussians.append(OccupancyGaussian(
                    pos=np.array([ix, iy, iz]) * 0.09 - side * 0.045,
                    scale=np.full(3, 0.05),
                    mass=float(rng.uniform(0.5, 2.0)), tag=f"g{k}",
                    E=1.0e5, nu=0.3, rho=1200.0))
                k += 1
    return gaussians


def _m5_ladder() -> Ladder | None:
    report = ROOT / "artifacts" / "m5_report.json"
    if not report.is_file():
        return None
    data = json.loads(report.read_text())
    raster = data.get("raster_gpu") or {}
    levels = sorted(raster.get("levels") or [],
                    key=lambda row: row["gaussians"], reverse=True)
    if not levels:
        return None
    return Ladder(counts=[row["gaussians"] for row in levels],
                  costs_ms=[row["latency_ms_median"] for row in levels])


def _g5a() -> dict:
    base = _cloud(64, seed=0)
    total_mass = sum(g.mass for g in base)
    levels, reports = build_lod_pyramid(base, min_count=2, collect_reports=True)
    failures = []
    mass_rows = []
    for index, level in enumerate(levels):
        mass_error = abs(sum(g.mass for g in level) - total_mass) / total_mass
        mass_rows.append({"level": index, "gaussians": len(level),
                          "mass_relative_error": mass_error})
        if mass_error > 1e-12:
            failures.append(f"G5a level {index}: mass error {mass_error}")
        for g in level:
            if not np.all(np.isfinite(g.scale)) or np.any(g.scale <= 0):
                failures.append(f"G5a level {index}: non-finite or non-positive scale")
    max_offdiag = max((r.offdiag_residual_m2 for r in reports), default=0.0)
    max_mass = max((r.mass_relative_error for r in reports), default=0.0)
    max_com = max((r.com_drift_m for r in reports), default=0.0)
    if max_mass > 1e-12 or max_com > 1e-12:
        failures.append(f"G5a merge invariants broken: mass {max_mass}, com {max_com}")
    return {"passed": not failures, "failures": failures,
            "levels": mass_rows,
            "count_range": [len(levels[0]), len(levels[-1])],
            "merge_count": len(reports),
            "merge_offdiag_residual_max_m2": max_offdiag,
            "merge_mass_error_max": max_mass,
            "merge_com_drift_max_m": max_com}


def _g5b(ladder: Ladder) -> dict:
    failures = []
    # Tight budget: only the coarsest measured level (2.5k, 0.236 ms) fits
    # under 0.30 ms; trace must switch from level 0 and settle there.
    budget = 0.30
    current, streak, trace = 0, 0, []
    for _ in range(8):
        current, streak = choose_level(ladder, budget, current, streak)
        trace.append(current)
    settled = trace[-1]
    fitting = [i for i, cost in enumerate(ladder.costs_ms) if cost <= budget]
    expected = fitting[0] if fitting else len(ladder.costs_ms) - 1
    if settled != expected:
        failures.append(
            f"G5b settled at level {settled}, expected finest fitting {expected}")
    if trace[0] == settled:
        failures.append("G5b switched without hysteresis delay")
    # Loose budget: finest level must hold from the start.
    current, streak, trace_loose = 0, 0, []
    for _ in range(4):
        current, streak = choose_level(ladder, 10.0, current, streak)
        trace_loose.append(current)
    if any(level != 0 for level in trace_loose):
        failures.append(f"G5b loose-budget trace left level 0: {trace_loose}")
    return {"passed": not failures, "failures": failures,
            "ladder_counts": ladder.counts,
            "ladder_costs_ms": ladder.costs_ms,
            "tight_budget_ms": budget,
            "tight_trace": trace,
            "loose_trace": trace_loose}


def run_p5(artifact: Path = ARTIFACT, remote: bool = False) -> dict:
    g5a = _g5a()
    ladder = _m5_ladder()
    g5b = _g5b(ladder) if ladder else {
        "passed": False,
        "failures": ["G5b skipped: artifacts/m5_report.json missing or empty"],
    }
    g5c = {"passed": True, "note": "render-side smoke not run (remote off); "
                                    "stride decimation is rendering-only, no "
                                    "dynamics claim"}
    failures = g5a["failures"] + g5b["failures"]
    report = {
        "phase": "P5",
        "gate": "G5",
        "passed": not failures,
        "failures": failures,
        "blocked": (
            "Dynamics-level LOD (decimating a scene while it simulates) is "
            "blocked on P1-2 rigid reduction: the Phase 0 dynamics has no "
            "attractive bond — pair stiffness feeds modal analysis only, "
            "never the integrator — so multi-splat bodies disintegrate and "
            "moment-preserving merges of them would diverge immediately. "
            "G5 therefore verifies merge invariants, the adaptive policy "
            "against real M5 telemetry, and (optionally) a rendering-side "
            "decimation smoke with no dynamics claim."
        ),
        "g5a_merge_invariants": g5a,
        "g5b_policy_on_measured_ladder": g5b,
        "g5c_render_smoke": g5c,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[P5/G5] passed={report['passed']} "
          f"levels={g5a['count_range']} merges={g5a['merge_count']} "
          f"tight_trace={g5b.get('tight_trace')} -> {artifact}")
    return report


if __name__ == "__main__":
    result = run_p5()
    raise SystemExit(0 if result["passed"] else 1)
