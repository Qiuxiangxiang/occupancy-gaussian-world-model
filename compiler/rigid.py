"""P1-2 rigid reduction: freeze a cluster's internal DoFs, keep translation.

Why this exists (recorded): the Phase 0 dynamics has no attractive bond —
pair stiffness feeds modal analysis only, never the integrator. A cluster
of splats is free particles whose internal overlap repels; two splats
overlapped by 0.01 m separate within ~19 steps. Rigid reduction is the
minimal constraint that makes multi-splat objects meaningful in dynamics.

Recorded decisions (glossary of record: tests/test_rigid.py):

- Constraint: each member keeps its rest offset from the group's center of
  mass. ``x_i = X_cm + d_i`` with d_i fixed at bind time. There is no
  rotational DOF on ``OccupancyGaussian`` (M2 note), so rotation is not
  modeled; rigid here means rigid translation.
- Enforcement: post-step projection. Members integrate with their own
  masses through ``langevin_step``, then positions/velocities are
  re-projected onto the constraint manifold: X_cm and V_cm are the
  mass-weighted means, members get ``X_cm + d_i`` and ``V_cm``. Linear
  momentum of the group is conserved exactly by construction.
- Cost, recorded honestly: internal kinetic energy is removed at every
  projection. Forces applied at different member points can excite
  internal tendencies; projection discards them. This is the rigid-body
  approximation, not an exact constrained dynamics.
- Binding is explicit: a group is declared by member indices; nothing
  infers rigidity from overlap or tags.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigidGroup:
    members: tuple[int, ...]
    offsets: np.ndarray   # (len(members), 3) rest offsets from the cm
    total_mass: float

    @classmethod
    def bind(cls, masses: np.ndarray, members) -> "RigidGroup":
        members = tuple(int(i) for i in members)
        if len(members) < 2:
            raise ValueError("a rigid group needs at least two members")
        if len(set(members)) != len(members):
            raise ValueError("rigid group members must be distinct")
        if any(i < 0 or i >= len(masses) for i in members):
            raise ValueError("rigid group member index out of range")
        return cls(members=members,
                   offsets=np.zeros((len(members), 3)),
                   total_mass=float(sum(masses[i] for i in members)))


def bind_groups(state_pos: np.ndarray, masses: np.ndarray,
                group_specs) -> list[RigidGroup]:
    """Build groups with rest offsets taken from the current positions."""
    groups = []
    for spec in group_specs:
        group = RigidGroup.bind(masses, spec)
        cm = _cm(state_pos, masses, group.members)
        for row, i in enumerate(group.members):
            group.offsets[row] = state_pos[i] - cm
        groups.append(group)
    return groups


def _cm(pos, masses, members):
    return sum(masses[i] * pos[i] for i in members) / sum(masses[i] for i in members)


def project_rigid(state, masses, groups: list[RigidGroup]):
    """Project positions and velocities onto the rigid manifolds, in place.

    Members of each group get the group's cm position plus the rest offset
    and the group's cm velocity. Non-member state is untouched.
    """
    pos = np.asarray(state.pos, dtype=np.float64)
    vel = np.asarray(state.vel, dtype=np.float64)
    for group in groups:
        m = np.array([masses[i] for i in group.members])
        rows = list(group.members)
        cm_pos = (m[:, None] * pos[rows]).sum(axis=0) / m.sum()
        cm_vel = (m[:, None] * vel[rows]).sum(axis=0) / m.sum()
        for row, offset in zip(rows, group.offsets):
            pos[row] = cm_pos + offset
            vel[row] = cm_vel
    state.pos = pos
    state.vel = vel
    return state


def group_cm(state, masses, group: RigidGroup) -> np.ndarray:
    return _cm(np.asarray(state.pos), masses, group.members)
