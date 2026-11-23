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
import sys

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


REMOTE_PYTHON = "/data/Users/AO/eacomp/liuqinghua/m5/venv/bin/python"
REMOTE_REPO = "/data/Users/AO/eacomp/liuqinghua/m5/repo"


def _g5c(host: str = "ea-4090-ts") -> dict:
    """Render-side LOD smoke on the 4090. Stride-decimated levels of the M5
    cloud each render one frame; coverage is reported, rendering without
    error and every level staying visible are gated."""
    import subprocess
    import tempfile
    from compiler.m5_telemetry import RASTER_LEVELS, generate_cloud
    ply = ROOT / "artifacts" / "m5_lod_input.ply"
    if not ply.is_file():
        generate_cloud(ply, 100_000, seed=0)
    sync = subprocess.run(
        ["rsync", "-az", "--exclude", ".git", "--exclude", "__pycache__",
         "--exclude", ".pytest_cache", "-e", "ssh -o BatchMode=yes",
         str(ROOT) + "/", f"{host}:{REMOTE_REPO}/"],
        capture_output=True, text=True, timeout=600)
    if sync.returncode != 0:
        return {"passed": False,
                "failures": [f"G5c rsync failed: {sync.stderr.strip()[:200]}"]}
    remote_ply = f"{REMOTE_REPO}/artifacts/m5_lod_input.ply"
    put = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                          f"mkdir -p {REMOTE_REPO}/artifacts"],
                         capture_output=True, text=True, timeout=60)
    cp = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(ply),
                         f"{host}:{remote_ply}"],
                        capture_output=True, text=True, timeout=600)
    if cp.returncode != 0:
        return {"passed": False,
                "failures": [f"G5c ply upload failed: {cp.stderr.strip()[:200]}"]}
    out = f"{REMOTE_REPO}/artifacts/p5_render_smoke.json"
    cmd = (f"cd {REMOTE_REPO} && {REMOTE_PYTHON} scripts/p5_render_smoke.py "
           f"--ply {remote_ply} --output {out} "
           f"--levels {' '.join(str(v) for v in RASTER_LEVELS)}")
    run = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                         capture_output=True, text=True, timeout=1800)
    if run.returncode != 0:
        return {"passed": False,
                "failures": [f"G5c remote render failed: {run.stderr.strip()[:300]}"]}
    local = ROOT / "artifacts" / "p5_render_smoke.json"
    fetch = subprocess.run(["scp", "-q", "-o", "BatchMode=yes",
                            f"{host}:{out}", str(local)],
                           capture_output=True, text=True, timeout=120)
    if fetch.returncode != 0:
        return {"passed": False,
                "failures": ["G5c report fetch failed"]}
    data = json.loads(local.read_text())
    failures = []
    rows = data.get("levels") or []
    if [row["gaussians"] for row in rows] != RASTER_LEVELS:
        failures.append(f"G5c rendered levels {[r['gaussians'] for r in rows]} "
                        f"!= requested {RASTER_LEVELS}")
    invisible = [row["gaussians"] for row in rows
                 if row["coverage_nonblack_fraction"] <= 0.01]
    if invisible:
        failures.append(f"G5c levels with no visible coverage: {invisible}")
    coverages = [row["coverage_nonblack_fraction"] for row in rows]
    return {"passed": not failures, "failures": failures,
            "gpu": data.get("gpu"),
            "levels": rows,
            "coverage_trace": coverages,
            "monotonicity": ("recorded, not gated: coverage "
                             + ("decreases with decimation"
                                if coverages == sorted(coverages, reverse=True)
                                else "is not monotone"))}


def run_p5(artifact: Path = ARTIFACT, remote: bool = False) -> dict:
    g5a = _g5a()
    ladder = _m5_ladder()
    g5b = _g5b(ladder) if ladder else {
        "passed": False,
        "failures": ["G5b skipped: artifacts/m5_report.json missing or empty"],
    }
    if remote:
        g5c = _g5c()
    else:
        g5c = {"passed": True, "note": "render-side smoke not run (remote off); "
                                       "stride decimation is rendering-only, no "
                                       "dynamics claim"}
    g5d = _g5d()
    failures = g5a["failures"] + g5b["failures"] + g5d["failures"]
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
        "g5d_dynamics_lod_equivalence": g5d,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[P5/G5] passed={report['passed']} "
          f"levels={g5a['count_range']} merges={g5a['merge_count']} "
          f"tight_trace={g5b.get('tight_trace')} -> {artifact}")
    return report


def _g5d() -> dict:
    """Dynamics-level LOD equivalence, unblocked by P1-2 rigid reduction.

    Fine scene: two rigid bodies of two splats each (1-sigma bonded along
    y, approach along x). Coarse scene: each body replaced by its
    moment-preserving merge. Both run the same M1-style approach; the
    comparison is PER-BODY center-of-mass trajectories (whole-scene cm is
    identically zero by symmetry and proves nothing). The deviation is the
    measured accuracy cost of the LOD switch — recorded, with contact
    event preservation, exact mass conservation, and a recorded tolerance
    (5% of total travel) gated.
    """
    from compiler.lod import merge_pair
    from compiler.rigid import bind_groups, group_cm, project_rigid
    from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                              contact_pairs, langevin_step,
                                              penalty_force)
    dt = 1.0e-4
    horizon_s = 0.4
    speed = 0.1
    fine = []
    for body, x0 in (("L", -0.070), ("R", 0.070)):
        for k, dy in enumerate((-0.045, 0.045)):
            fine.append(OccupancyGaussian(
                pos=np.array([x0, dy, 0.0]), scale=np.ones(3) * 0.05,
                mass=2.36, tag=f"{body}{k}", E=1.0e5, nu=0.3, rho=1200.0))
    coarse = [merge_pair(fine[0], fine[1]), merge_pair(fine[2], fine[3])]
    # Body identity for the cross-contact counter: merged tags would both
    # start with "(" and never compare cross. Restore explicit body tags.
    coarse[0].tag = "L"
    coarse[1].tag = "R"

    def simulate(gaussians, rigid_specs):
        n = len(gaussians)
        masses = np.array([g.mass for g in gaussians])
        state = State(pos=np.stack([g.pos.copy() for g in gaussians]),
                      vel=np.zeros((n, 3)))
        state.vel[:, 0] = [speed if g.pos[0] < 0 else -speed for g in gaussians]
        groups = (bind_groups(state.pos, masses, rigid_specs)
                  if rigid_specs else [])
        steps = int(round(horizon_s / dt))
        trace = []
        contact_steps = 0
        for _ in range(steps):
            for i in range(n):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            cross = any(gaussians[p.i].tag[0] != gaussians[p.j].tag[0]
                        for p in pairs)
            if cross:
                contact_steps += 1
            force = np.zeros((n, 3))
            for pair in pairs:
                fi, fj = penalty_force(pair, gaussians, mu=0.0)
                force[pair.i] += fi
                force[pair.j] += fj

            def force_fn(pos, vel=state.vel.copy()):
                for i in range(n):
                    gaussians[i].pos = np.array(pos[i], dtype=np.float64, copy=True)
                f = np.zeros((n, 3))
                for pair in contact_pairs(gaussians, velocities=vel, threshold=0.0):
                    fi, fj = penalty_force(pair, gaussians, mu=0.0)
                    f[pair.i] += fi
                    f[pair.j] += fj
                return f

            state = langevin_step(state, force, dt, masses, T=0.0, gamma=0.0,
                                  force_fn=force_fn)
            if groups:
                project_rigid(state, masses, groups)
            if groups:
                trace.append([group_cm(state, masses, groups[0]),
                              group_cm(state, masses, groups[1])])
            else:
                trace.append([state.pos[0].copy(), state.pos[1].copy()])
        return trace, contact_steps

    fine_trace, fine_contacts = simulate(
        [type(g)(pos=g.pos.copy(), scale=g.scale.copy(), mass=g.mass, tag=g.tag,
                 E=g.E, nu=g.nu, rho=g.rho) for g in fine],
        [(0, 1), (2, 3)])
    coarse_trace, coarse_contacts = simulate(
        [type(g)(pos=g.pos.copy(), scale=g.scale.copy(), mass=g.mass, tag=g.tag,
                 E=g.E, nu=g.nu, rho=g.rho) for g in coarse],
        None)

    fine_cm = np.array([[row[0], row[1]] for row in fine_trace])  # (steps, 2, 3)
    coarse_cm = np.array([[row[0], row[1]] for row in coarse_trace])
    deviation = np.abs(fine_cm - coarse_cm).max(axis=(0, 2))  # per body
    travel = 2 * speed * horizon_s
    failures = []
    if fine_contacts == 0:
        failures.append("G5d: fine scene had no contact event")
    if coarse_contacts == 0:
        failures.append("G5d: coarse scene had no contact event")
    mass_fine = sum(g.mass for g in fine)
    mass_coarse = sum(g.mass for g in coarse)
    mass_rel = abs(mass_coarse - mass_fine) / mass_fine
    if mass_rel > 1e-12:
        failures.append("G5d: mass not conserved by the merge")
    deviation_tolerance = 0.05 * travel
    if deviation[0] > deviation_tolerance or deviation[1] > deviation_tolerance:
        failures.append(
            f"G5d: per-body cm deviation {deviation.tolist()} m exceeds "
            f"recorded tolerance {deviation_tolerance:.4f} m")
    return {"passed": not failures, "failures": failures,
            "horizon_s": horizon_s, "approach_speed_each": speed,
            "contact_steps": {"fine": fine_contacts, "coarse": coarse_contacts},
            "cm_deviation_max_per_body_m": [float(v) for v in deviation],
            "cm_deviation_tolerance_m": float(deviation_tolerance),
            "tolerance_basis": "5% of total cm travel, recorded decision",
            "mass_relative_error": mass_rel}


if __name__ == "__main__":
    result = run_p5(remote="--remote" in sys.argv)
    raise SystemExit(0 if result["passed"] else 1)
