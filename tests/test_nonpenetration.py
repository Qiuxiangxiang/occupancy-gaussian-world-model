"""P1-3 non-penetration iteration — glossary of record (before the physics).

Recorded decisions:

- What: after each integrator step, active contact pairs with penetration
  > 0 are moved apart along the pair normal, inverse-mass weighted, for up
  to K iterations (K recorded, default 8), stopping early when the maximum
  penetration falls below a recorded tolerance (default 1e-6 m).
- Split rule: pair displacement pen is shared as
  dx_i = +pen * (m_j / (m_i + m_j)) * n and
  dx_j = -pen * (m_i / (m_i + m_j)) * n. Per pair this conserves linear
  momentum exactly (m_i dx_i + m_j dx_j = 0); the whole pass therefore
  conserves total momentum exactly.
- Velocities are not touched by the position pass. The removed kinetic
  energy is accounted, not hidden: work = sum over pairs of the penalty
  force dotted with the displacement applied, accumulated per step and
  reported as heat. This is the first entry of the P1-4 energy split
  (kinetic / potential / heat); friction and damping heat are still open.
- Rigid groups: the non-penetration pass runs BEFORE the rigid projection,
  and the pair (contact pass, rigid pass) is iterated together when both
  are active — rigid re-projection can re-introduce penetration, one pass
  alone cannot satisfy both manifolds. Iteration count is recorded.
- Honesty boundary: this is a penalty + projection method, not a
  complementarity solver. Gap/force complementarity (M3 axis) is still
  measurement only; P1-3 narrows penetration, it does not claim
  non-penetration in the complementarity sense.
"""
import numpy as np
import pytest

from compiler.nonpenetration import resolve_penetration
from compiler.rigid import bind_groups, project_rigid
from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                          contact_pairs, langevin_step,
                                          penalty_force)

DT = 1.0e-4


def overlapped_pair(pen=0.02):
    """Two splats along x, center distance = reach - pen (overlapped)."""
    dist = 0.10 - pen
    gaussians = [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, tag="A"),
        OccupancyGaussian(pos=np.array([+dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=3.0, tag="B"),
    ]
    return gaussians


def test_resolve_removes_penetration_and_conserves_momentum():
    gaussians = overlapped_pair(pen=0.02)
    masses = np.array([1.0, 3.0])
    pos0 = np.stack([g.pos for g in gaussians])
    state = State(pos=pos0.copy(),
                  vel=np.array([[0.1, 0, 0], [-0.1, 0, 0]]))
    momentum0 = (masses[:, None] * state.vel).sum(axis=0)
    pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
    work = resolve_penetration(state, gaussians, pairs, masses,
                               iterations=8, tol_m=1e-9)
    remaining = contact_pairs(gaussians, threshold=0.0)
    assert remaining == [] or all(p.penetration <= 1e-9 for p in remaining)
    # Inverse-mass split: the light splat moves more.
    moved = state.pos - pos0
    assert np.linalg.norm(moved[0]) > np.linalg.norm(moved[1])
    momentum1 = (masses[:, None] * state.vel).sum(axis=0)
    assert np.allclose(momentum1, momentum0, atol=1e-15)
    assert work > 0.0  # displaced against the penalty force: heat accounted


def test_resolve_reports_zero_work_when_nothing_moves():
    gaussians = overlapped_pair(pen=0.0)
    # Pull them apart first so no pair is active.
    gaussians[0].pos = gaussians[0].pos - np.array([0.2, 0, 0])
    gaussians[1].pos = gaussians[1].pos + np.array([0.2, 0, 0])
    masses = np.array([1.0, 3.0])
    state = State(pos=np.stack([g.pos for g in gaussians]),
                  vel=np.zeros((2, 3)))
    pairs = contact_pairs(gaussians, threshold=0.0)
    work = resolve_penetration(state, gaussians, pairs, masses)
    assert work == 0.0


def test_hard_collision_penetration_reduced_with_iteration():
    """M1-style head-on splats at high speed: max penetration with the
    non-penetration pass must be strictly lower than without, at the same
    dt; the heat accumulated is reported."""
    def run(resolve: bool):
        gaussians = [
            OccupancyGaussian(pos=np.array([-0.055, 0.0, 0.0]),
                              scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
            OccupancyGaussian(pos=np.array([+0.055, 0.0, 0.0]),
                              scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
        ]
        masses = np.array([2.36, 2.36])
        state = State(pos=np.stack([g.pos for g in gaussians]),
                      vel=np.array([[0.3, 0, 0], [-0.3, 0, 0]]))
        max_pen, heat = 0.0, 0.0
        for _ in range(int(round(0.3 / DT))):
            for i in range(2):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            force = np.zeros((2, 3))
            for pair in pairs:
                fi, fj = penalty_force(pair, gaussians, mu=0.0)
                force[pair.i] += fi
                force[pair.j] += fj

            def force_fn(pos, vel=state.vel.copy()):
                for i in range(2):
                    gaussians[i].pos = np.array(pos[i], dtype=np.float64, copy=True)
                f = np.zeros((2, 3))
                for pair in contact_pairs(gaussians, velocities=vel, threshold=0.0):
                    fi, fj = penalty_force(pair, gaussians, mu=0.0)
                    f[pair.i] += fi
                    f[pair.j] += fj
                return f

            state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0,
                                  force_fn=force_fn)
            for i in range(2):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            for pair in pairs:
                max_pen = max(max_pen, pair.penetration)
            if resolve and pairs:
                heat += resolve_penetration(state, gaussians, pairs, masses,
                                            iterations=8, tol_m=1e-9)
        return max_pen, heat

    pen_free, _ = run(False)
    pen_fixed, heat = run(True)
    assert pen_fixed < pen_free
    assert heat > 0.0


def test_composes_with_rigid_projection():
    """Rigid ring stays rigid when the non-penetration pass runs before the
    projection; the pair of passes is iterated together (recorded)."""
    from scenes.cup_and_cup import build_scene, DT
    gaussians, group_specs = build_scene()
    n = len(gaussians)
    masses = np.array([g.mass for g in gaussians])
    state = State(pos=np.stack([g.pos for g in gaussians]),
                  vel=np.zeros((n, 3)))
    state.vel[:, 0] = [0.1 if g.pos[0] < 0 else -0.1 for g in gaussians]
    groups = bind_groups(state.pos, masses, group_specs)
    separation0 = _separations(state.pos, masses, groups)
    for _ in range(200):
        for i in range(n):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        force = np.zeros((n, 3))
        for pair in pairs:
            fi, fj = penalty_force(pair, gaussians, mu=0.0)
            force[pair.i] += fi
            force[pair.j] += fj
        state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0)
        # Iterated together: contact pass then rigid pass, twice (recorded).
        for _ in range(2):
            for i in range(n):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            if pairs:
                resolve_penetration(state, gaussians, pairs, masses,
                                    iterations=4, tol_m=1e-9)
            project_rigid(state, masses, groups)
    separation1 = _separations(state.pos, masses, groups)
    assert np.max(np.abs(separation1 - separation0)) < 1e-12


def _separations(pos, masses, groups):
    rows = []
    for group in groups:
        members = list(group.members)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                rows.append(np.linalg.norm(pos[members[a]] - pos[members[b]]))
    return np.array(rows)
