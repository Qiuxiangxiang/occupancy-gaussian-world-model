"""P1-3 non-penetration iteration: position projection for contact pairs.

Glossary of record: tests/test_nonpenetration.py. Summary: active pairs
with penetration are moved apart along the pair normal, inverse-mass
weighted, up to K iterations; per-pair momentum conservation is exact;
velocities are untouched; the penalty work removed is accumulated and
reported as heat (first entry of the P1-4 energy split). This narrows
penetration — it is not a complementarity solver, and no such claim is
made (M3 axis stays measurement-only).
"""
from __future__ import annotations

import numpy as np

from occupancy_gaussian_framework import contact_pairs, pairwise_stiffness


def resolve_penetration(state, gaussians, pairs, masses,
                        *, iterations: int = 8, tol_m: float = 1e-6) -> float:
    """Move overlapping pairs apart. Returns the accounted heat: the sum of
    penalty force dotted with displacement, i.e. the work removed from the
    mechanical energy by the projection."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    heat = 0.0
    pos = np.asarray(state.pos, dtype=np.float64)
    active = list(pairs)
    for _ in range(iterations):
        worst = 0.0
        for pair in active:
            pen = float(pair.penetration)
            # Pairs within tolerance are touching, not penetrating: no
            # displacement and no heat. Charging them would manufacture
            # heat every step at sustained contact (k_n is huge, so even
            # boundary noise pen ~ 1e-7 m books real joules).
            if pen <= tol_m:
                continue
            i, j = pair.i, pair.j
            mi, mj = float(masses[i]), float(masses[j])
            total = mi + mj
            n = np.asarray(pair.normal, dtype=np.float64)
            # n points from i to j; overlapping means moving i along +n
            # deepens the overlap. Separate: i backs along -n, j along +n.
            pos[i] -= pen * (mj / total) * n
            pos[j] += pen * (mi / total) * n
            # Accounted heat: work of the penalty force over the removal
            # path. The force at penetration x is k*x; removing the
            # overlap travels x from pen to 0, so the work (and the PE
            # removed) is integral k*x dx = 0.5*k*pen^2.
            k_n = float(pair.normal @ pairwise_stiffness(
                gaussians[i], gaussians[j]) @ pair.normal)
            heat += 0.5 * k_n * pen * pen
            worst = max(worst, pen)
        if worst <= tol_m:
            break
        # Refresh the pair list against the moved positions; new overlaps
        # created by the sweep are picked up, resolved ones drop out.
        for k, g in enumerate(gaussians):
            g.pos = pos[k].copy()
        active = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
    state.pos = pos
    for k, g in enumerate(gaussians):
        g.pos = pos[k].copy()
    return heat
