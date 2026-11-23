"""P1-2 rigid reduction — glossary of record (written before the physics).

Recorded decisions:

- Constraint: members keep fixed rest offsets from the group's center of
  mass; ``x_i = X_cm + d_i``. No rotational DOF exists on
  ``OccupancyGaussian`` (M2 note), so rigid means rigid translation only.
- Enforcement: post-step projection after the members' own ``langevin_step``
  (mass-weighted cm position/velocity; members receive ``X_cm + d_i``,
  ``V_cm``). Group linear momentum is conserved exactly by construction.
- Approximation, stated: internal kinetic energy is removed at every
  projection. Forces at different member points excite internal tendencies
  that projection discards. This is the rigid-body approximation.
- Binding is explicit (member indices given by the caller); nothing infers
  rigidity from overlap or tags.
- Motivating failure, kept as a regression target: two splats overlapped
  0.01 m without a bond separate within ~19 steps; the same pair bound
  rigidly must keep its separation to numerical precision for the same
  horizon.
"""
import numpy as np
import pytest

from compiler.rigid import bind_groups, group_cm, project_rigid
from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                          contact_pairs, langevin_step,
                                          penalty_force)

DT = 1.0e-4
HORIZON_STEPS = 4000


def bonded_pair(overlap=0.01):
    """Two splats along y, center distance 0.09 (0.01 m overlapped at
    1-sigma), plus a distant spectator so contact bookkeeping is honest."""
    gaussians = [
        OccupancyGaussian(pos=np.array([0.0, -0.045, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, tag="A0"),
        OccupancyGaussian(pos=np.array([0.0, 0.045, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, tag="A1"),
        OccupancyGaussian(pos=np.array([5.0, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, tag="S"),
    ]
    return gaussians


def run(state, gaussians, masses, groups, steps=HORIZON_STEPS):
    n = len(gaussians)
    for _ in range(steps):
        for i in range(n):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
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
    return state


def test_unbonded_pair_disintegrates_within_19_steps():
    gaussians = bonded_pair()
    masses = np.ones(3)
    state = State(pos=np.stack([g.pos for g in gaussians]), vel=np.zeros((3, 3)))
    initial = np.linalg.norm(state.pos[0] - state.pos[1])
    separated_at = None
    for step in range(HORIZON_STEPS):
        for i in range(3):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, threshold=0.0)
        force = np.zeros((3, 3))
        for pair in pairs:
            fi, fj = penalty_force(pair, gaussians, mu=0.0)
            force[pair.i] += fi
            force[pair.j] += fj
        state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0)
        if np.linalg.norm(state.pos[0] - state.pos[1]) > initial + 1e-6:
            separated_at = step
            break
    assert separated_at is not None and separated_at <= 19


def test_bonded_pair_keeps_separation_and_momentum():
    gaussians = bonded_pair()
    masses = np.ones(3)
    state = State(pos=np.stack([g.pos for g in gaussians]),
                  vel=np.array([[0.05, 0, 0], [0.05, 0, 0], [0.0, 0, 0]]))
    groups = bind_groups(state.pos, masses, [(0, 1)])
    separation0 = np.linalg.norm(state.pos[0] - state.pos[1])
    momentum0 = masses[0] * state.vel[0] + masses[1] * state.vel[1]
    state = run(state, gaussians, masses, groups)
    separation1 = np.linalg.norm(state.pos[0] - state.pos[1])
    momentum1 = masses[0] * state.vel[0] + masses[1] * state.vel[1]
    assert separation1 == pytest.approx(separation0, abs=1e-12)
    assert np.allclose(momentum1, momentum0, atol=1e-12)
    # The bonded pair translates coherently: both members share the cm.
    cm = group_cm(state, masses, groups[0])
    assert np.allclose(state.pos[0], cm + groups[0].offsets[0], atol=1e-12)
    assert np.allclose(state.pos[1], cm + groups[0].offsets[1], atol=1e-12)


def test_internal_kinetic_energy_is_removed_by_projection():
    masses = np.array([1.0, 1.0])
    state = State(pos=np.array([[0.0, -0.045, 0.0], [0.0, 0.045, 0.0]]),
                  vel=np.array([[0.0, 0.3, 0.0], [0.0, -0.3, 0.0]]))
    groups = bind_groups(state.pos, masses, [(0, 1)])
    project_rigid(state, masses, groups)
    # cm velocity is the mean (zero here); internal motion is gone.
    assert np.allclose(state.vel, 0.0, atol=1e-15)
    assert np.linalg.norm(state.pos[0] - state.pos[1]) == pytest.approx(0.09)


def test_binding_requires_two_members_and_rejects_unknown_index():
    masses = np.ones(3)
    pos = np.zeros((3, 3))
    with pytest.raises(ValueError):
        bind_groups(pos, masses, [(0,)])
    with pytest.raises(ValueError):
        bind_groups(pos, masses, [(0, 7)])
    with pytest.raises(ValueError):
        bind_groups(pos, masses, [(0, 0)])
