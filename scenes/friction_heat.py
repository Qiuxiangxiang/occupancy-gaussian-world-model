"""P1-4 friction heat: tangential collision with Coulomb friction.

Glossary of record: tests/test_friction_heat.py. Two splats collide
head-on along x while sliding oppositely along y, so the Coulomb term in
``penalty_force`` dissipates tangential kinetic energy. The heat is
accounted per step as ``-(f_t dot v_rel_t) * dt`` and the closure
``KE_end + heat + PE_end == KE_0`` is the gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                          contact_pairs, langevin_step,
                                          penalty_force, pairwise_stiffness)

DT = 1.0e-4
HORIZON_S = 0.3
APPROACH_SPEED = 0.2
SLIDE_SPEED = 0.1

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "friction_heat_report.json"


def build():
    gaussians = [
        OccupancyGaussian(pos=np.array([-0.055, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
        OccupancyGaussian(pos=np.array([0.055, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
    ]
    return gaussians


def _pair_quantities(pair, gaussians, mu):
    """Tangential (friction) force on body i and tangential relative
    velocity. Heat rate = f_t_i dot v_rel_t: friction on i pulls along the
    relative motion of j, and the pair work is -that (equal and opposite
    forces), so the dissipative heat is +f_t_i dot v_rel_t."""
    fi, _ = penalty_force(pair, gaussians, mu=mu)
    n = np.asarray(pair.normal, dtype=np.float64)
    f_i = np.asarray(fi, dtype=np.float64)
    f_t = f_i - np.dot(f_i, n) * n
    v_rel = np.asarray(pair.rel_vel, dtype=np.float64)
    v_rel_t = v_rel - np.dot(v_rel, n) * n
    return f_t, v_rel_t


def collide(mu: float, steps: int | None = None) -> dict:
    gaussians = build()
    masses = np.array([2.36, 2.36])
    state = State(pos=np.stack([g.pos for g in gaussians]),
                  vel=np.array([[APPROACH_SPEED, SLIDE_SPEED, 0.0],
                                [-APPROACH_SPEED, -SLIDE_SPEED, 0.0]]))
    kinetic0 = 0.5 * float(np.sum(masses[:, None] * state.vel ** 2))
    momentum0 = (masses[:, None] * state.vel).sum(axis=0)
    n_steps = steps or int(round(HORIZON_S / DT))
    heat = 0.0
    contact_steps = 0
    potential_end = 0.0
    for _ in range(n_steps):
        for i in range(2):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        force = np.zeros((2, 3))
        for pair in pairs:
            fi, fj = penalty_force(pair, gaussians, mu=mu)
            force[pair.i] += fi
            force[pair.j] += fj
            if mu > 0.0:
                f_t, v_rel_t = _pair_quantities(pair, gaussians, mu)
                heat += float(np.dot(f_t, v_rel_t)) * DT
        if pairs:
            contact_steps += 1

        def force_fn(pos, vel=state.vel.copy()):
            for i in range(2):
                gaussians[i].pos = np.array(pos[i], dtype=np.float64, copy=True)
            f = np.zeros((2, 3))
            for pair in contact_pairs(gaussians, velocities=vel, threshold=0.0):
                fi, fj = penalty_force(pair, gaussians, mu=mu)
                f[pair.i] += fi
                f[pair.j] += fj
            return f

        state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0,
                              force_fn=force_fn)
    for i in range(2):
        gaussians[i].pos = state.pos[i].copy()
    pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
    for pair in pairs:
        k_n = float(pair.normal @ pairwise_stiffness(
            gaussians[pair.i], gaussians[pair.j]) @ pair.normal)
        potential_end += 0.5 * k_n * float(pair.penetration) ** 2
    kinetic_end = 0.5 * float(np.sum(masses[:, None] * state.vel ** 2))
    momentum_end = (masses[:, None] * state.vel).sum(axis=0)
    rel = state.vel[1] - state.vel[0]
    tangential_end = float(np.linalg.norm(rel[1:]))
    return {"mu": mu, "dt_s": DT, "steps": n_steps,
            "contact_steps": contact_steps,
            "kinetic_start_j": kinetic0,
            "kinetic_end_j": kinetic_end,
            "potential_end_j": potential_end,
            "friction_heat_j": heat,
            "closure_residual_j": kinetic_end + heat + potential_end - kinetic0,
            "momentum_drift_max": float(np.max(np.abs(momentum_end - momentum0))),
            "tangential_rel_speed_end": tangential_end}


def glossary_check():
    """The sign convention is part of the glossary: friction heat must be
    non-negative for any sliding contact; a convention that yields negative
    heat for the reference collision contradicts the glossary."""
    result = collide(mu=0.5, steps=1500)
    assert result["friction_heat_j"] >= 0.0


if __name__ == "__main__":
    out = {"mu_0.5": collide(mu=0.5), "mu_0.0": collide(mu=0.0)}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
