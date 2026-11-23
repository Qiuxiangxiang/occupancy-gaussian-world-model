"""Cup-and-cup collision scene: two rigid rings of splats meet head-on.

The glossary of record is tests/test_cup_scene.py. Run directly for the
metrics report; used by tests/test_cup_scene.py for the gates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as C
from compiler.rigid import bind_groups, project_rigid
from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                          contact_pairs, eval_energy_drift,
                                          langevin_step, pairwise_stiffness,
                                          penalty_force)

N_RING = 8
CUP_RADIUS = 0.05
CUP_SCALE = 0.02
APPROACH_SPEED = 0.1
DT = 1.0e-4
HORIZON_S = 0.4

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "cup_and_cup_report.json"


def build_cup(x0: float, tag: str, phase: float = 0.0) -> list[OccupancyGaussian]:
    """Ring of N_RING splats in the yz-plane, centered at x0. ``phase``
    rotates the ring so the two cups meet staggered (splat-to-gap), not
    ring-plane flat on ring-plane."""
    gaussians = []
    for k in range(N_RING):
        angle = phase + 2 * np.pi * k / N_RING
        pos = np.array([x0,
                        CUP_RADIUS * np.cos(angle),
                        CUP_RADIUS * np.sin(angle)])
        gaussians.append(OccupancyGaussian(
            pos=pos, scale=np.ones(3) * CUP_SCALE, mass=0.5,
            tag=f"{tag}{k}", E=1.0e5, nu=0.3, rho=1200.0))
    return gaussians


def build_scene():
    """Two cups, centers separated along x, rings aligned in the yz plane."""
    gaussians = build_cup(-0.050, "L") + build_cup(0.050, "R", phase=np.pi / N_RING)
    n = N_RING
    group_specs = [tuple(range(0, n)), tuple(range(n, 2 * n))]
    return gaussians, group_specs


def simulate(steps: int | None = None) -> dict:
    gaussians, group_specs = build_scene()
    n = len(gaussians)
    masses = np.array([g.mass for g in gaussians])
    state = State(pos=np.stack([g.pos.copy() for g in gaussians]),
                  vel=np.zeros((n, 3)))
    state.vel[:, 0] = [APPROACH_SPEED if g.pos[0] < 0 else -APPROACH_SPEED
                       for g in gaussians]
    groups = bind_groups(state.pos, masses, group_specs)
    separation0 = _member_separations(state.pos, groups)
    momentum0 = (masses[:, None] * state.vel).sum(axis=0)
    total_mass = float(masses.sum())
    steps = steps or int(round(HORIZON_S / DT))
    kinetic, potential = [], []
    contact_steps = 0
    cross_pen = 0.0
    scale_min = CUP_SCALE
    for _ in range(steps):
        for i in range(n):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        if any(gaussians[p.i].tag[0] != gaussians[p.j].tag[0] for p in pairs):
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

        state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0,
                              force_fn=force_fn)
        project_rigid(state, masses, groups)
        for pair in contact_pairs(gaussians, velocities=state.vel, threshold=0.0):
            if gaussians[pair.i].tag[0] != gaussians[pair.j].tag[0]:
                cross_pen = max(cross_pen, pair.penetration / scale_min)
        kinetic.append(0.5 * float(np.sum(masses[:, None] * state.vel ** 2)))
        pot = 0.0
        for i in range(n):
            gaussians[i].pos = state.pos[i].copy()
        for pair in contact_pairs(gaussians, velocities=state.vel, threshold=0.0):
            k_n = float(pair.normal @ pairwise_stiffness(
                gaussians[pair.i], gaussians[pair.j]) @ pair.normal)
            pot += 0.5 * k_n * float(pair.penetration) ** 2
        potential.append(pot)

    momentum1 = (masses[:, None] * state.vel).sum(axis=0)
    separation1 = _member_separations(state.pos, groups)
    rigidity = float(np.max(np.abs(separation1 - separation0)))
    momentum_drift = float(np.max(np.abs(momentum1 - momentum0)))
    drift = float(eval_energy_drift(kinetic, potential, DT))
    final_rings = [_ring_radii(state.pos[i * N_RING:(i + 1) * N_RING], i)
                   for i in range(2)]
    failures = []
    if contact_steps == 0:
        failures.append("no cross-cup contact")
    if rigidity > 1e-12:
        failures.append(f"rigidity drift {rigidity}")
    if momentum_drift > 1e-12:
        failures.append(f"momentum drift {momentum_drift}")
    if cross_pen >= C.PEN_THRESHOLD:
        failures.append(f"cross penetration {cross_pen} over limit")
    if drift >= C.ENERGY_DRIFT_LIMIT:
        failures.append(f"energy drift {drift}/s over limit {C.ENERGY_DRIFT_LIMIT}")
    return {"passed": not failures, "failures": failures,
            "contact_steps": contact_steps,
            "cross_penetration_ratio": float(cross_pen),
            "energy_drift_per_s": drift,
            "momentum_drift_max": momentum_drift,
            "rigidity_drift_max": rigidity,
            "final_rings": final_rings,
            "final_cm_x": [float(state.pos[i * N_RING:(i + 1) * N_RING, 0].mean())
                           for i in range(2)],
            "dt_s": DT, "steps": steps, "T_K": 0.0, "gamma_per_s": 0.0,
            "approach_speed_each": APPROACH_SPEED}


def _member_separations(pos, groups):
    rows = []
    for group in groups:
        members = list(group.members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                rows.append(np.linalg.norm(pos[members[a]] - pos[members[b]]))
    return np.array(rows)


def _ring_radii(pos, cup_index):
    x0 = pos[:, 0].mean()
    return [float(np.linalg.norm([p[1], p[2]])) for p in pos]


def export_ply(path: Path) -> None:
    """Static frame of the initial scene in the Cross3DGS exchange format."""
    from compiler.cross3dgs_ply import cloud_from_activated, write_cross3dgs_ply
    gaussians, _ = build_scene()
    xyz = np.stack([g.pos for g in gaussians])
    scale = np.stack([g.scale for g in gaussians])
    alpha = np.full(len(gaussians), 0.5)
    quat = np.zeros((len(gaussians), 4))
    quat[:, 0] = 1.0
    f_dc = np.zeros((len(gaussians), 3))
    f_dc[:, 0] = 0.8   # reddish cups
    cloud = cloud_from_activated(xyz, scale, alpha, quat, f_dc)
    write_cross3dgs_ply(path, cloud)


if __name__ == "__main__":
    import sys
    result = simulate()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if "--ply" in sys.argv:
        out = ROOT / "artifacts" / "cup_and_cup.ply"
        export_ply(out)
        print(f"exported {out}")
    raise SystemExit(0 if result["passed"] else 1)
