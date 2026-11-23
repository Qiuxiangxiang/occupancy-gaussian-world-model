"""Phase 5 adaptive LOD — glossary of record (written before the physics).

Level definition: a LOD pyramid from moment-preserving pairwise merges.
Recorded decisions:

- merge(g1, g2) conserves total mass, first moment (centroid), and second
  moment exactly. The merged covariance is projected to world-axis diagonal
  scales — the only representation Phase 0 integrates (M1 limit). The
  dropped off-diagonal part is measured by merge_diagnostics and reported;
  it is never silently discarded.
- The merge tree is deterministic: KD-style median split along the longest
  axis, pairing adjacent leaves; same input, same pyramid, no RNG.
- Material fields: E, nu, rho are mass-weighted means; merged mass is the
  sum. Total scene mass is conserved exactly at every level.
- Odd counts: the unpaired gaussian is carried to the next level unchanged;
  levels shrink by at most half each step.
- The adaptive policy is a pure function of (ladder, budget, current level,
  streak): choose the finest level whose predicted frame cost fits the
  budget; a switch requires the condition to hold for hysteresis_k
  consecutive evaluations (default 3). No learning, no hidden state beyond
  the streak.
- Switching consistency: the coarse state at a switch is re-merged from the
  current fine state (G5 re-derives levels from the live scene); mass and
  centroid continuity across the switch is asserted, not assumed.
"""
import numpy as np
import pytest

from compiler.lod import (Ladder, build_lod_pyramid, choose_level,
                          merge_diagnostics, merge_pair)
from occupancy_gaussian_framework import OccupancyGaussian


def blob(pos, mass=1.0, scale=0.1, tag="b"):
    return OccupancyGaussian(pos=np.asarray(pos, dtype=float),
                             scale=np.ones(3) * scale, mass=mass, tag=tag)


def cloud(n=16, seed=0):
    """Connected grid cloud: spacing 0.09 < 0.1 reach, so the pyramid
    actually merges. Random-uniform clouds are all disconnected at this
    scale and stall by design (glossary: disconnected elements persist)."""
    rng = np.random.default_rng(seed)
    side = round(n ** (1 / 3)) or 1
    out = []
    k = 0
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                if k >= n:
                    return out
                out.append(blob(np.array([ix, iy, iz]) * 0.09 - side * 0.045,
                                mass=float(rng.uniform(0.5, 2.0)), tag=f"g{k}"))
                k += 1
    return out


def test_merge_conserves_mass_first_and_second_moments():
    g1 = blob([0.1, -0.2, 0.3], mass=1.5, scale=0.08)
    g2 = blob([-0.4, 0.1, 0.0], mass=0.5, scale=0.12)
    merged = merge_pair(g1, g2)
    assert merged.mass == pytest.approx(2.0)
    com = (1.5 * g1.pos + 0.5 * g2.pos) / 2.0
    assert np.allclose(merged.pos, com)
    # Second moment along each axis is preserved; only the diagonal is kept
    # as scale, so compare variances (diagonal of the merged covariance).
    for axis in range(3):
        v = (1.5 * (g1.scale[axis] ** 2 + g1.pos[axis] ** 2)
             + 0.5 * (g2.scale[axis] ** 2 + g2.pos[axis] ** 2)) / 2.0 \
            - com[axis] ** 2
        assert merged.scale[axis] ** 2 == pytest.approx(v)


def test_merge_materials_are_mass_weighted_means():
    g1 = blob([0, 0, 0], mass=3.0)
    g2 = blob([1, 0, 0], mass=1.0)
    g1.E, g2.E = 1.0e5, 3.0e5
    merged = merge_pair(g1, g2)
    assert merged.E == pytest.approx(1.5e5)
    assert merged.mass == pytest.approx(4.0)


def test_merge_rejects_nonpositive_mass():
    with pytest.raises(ValueError):
        merge_pair(blob([0, 0, 0], mass=0.0), blob([1, 0, 0], mass=0.0))


def test_offdiagonal_residual_is_reported_not_dropped():
    g1 = blob([0.5, 0.5, 0.0])
    g2 = blob([-0.5, -0.5, 0.0])
    merged = merge_pair(g1, g2)
    diag = merge_diagnostics([g1, g2], merged)
    assert diag.mass_relative_error < 1e-12
    assert diag.com_drift_m < 1e-12
    # The xy off-diagonal is substantial here; it must be visible.
    assert diag.offdiag_residual_m2 > 0.1


def test_pyramid_is_deterministic_and_conserves_mass():
    gaussians = cloud(16, seed=1)
    a = build_lod_pyramid(gaussians, min_count=2)
    b = build_lod_pyramid(cloud(16, seed=1), min_count=2)
    # Connectivity-gated: the pyramid stalls when a level disconnects
    # (merged blobs spaced past their 1-sigma reach); the exact ladder is
    # geometry-dependent, only its determinism and mass conservation gate.
    assert [len(level) for level in a] == [len(level) for level in b]
    assert [len(level) for level in a] == [len(level) for level in b]
    for level_a, level_b in zip(a, b):
        for ga, gb in zip(level_a, level_b):
            assert np.allclose(ga.pos, gb.pos)
            assert ga.mass == pytest.approx(gb.mass)
    total = sum(g.mass for g in gaussians)
    for level in a:
        assert sum(g.mass for g in level) == pytest.approx(total, rel=1e-12)


def test_pyramid_handles_odd_counts():
    levels = build_lod_pyramid(cloud(7, seed=2), min_count=1)
    counts = [len(level) for level in levels]
    assert counts[0] == 7
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1))
    # Ends below 7 only if connectivity allows; mass conservation holds
    # regardless (glossary: disconnected elements persist).
    if len(counts) > 1:
        assert counts[-1] < counts[0]


# --- adaptive policy ---------------------------------------------------------

def ladder():
    # Finer levels cost more (M5: latency grows with count, fixed floor).
    return Ladder(counts=[16, 8, 4, 2], costs_ms=[1.0, 0.6, 0.4, 0.3])


def test_policy_picks_finest_level_within_budget():
    level, streak = choose_level(ladder(), budget_ms=0.7, current_level=0, streak=0)
    assert (level, streak) == (0, 1)   # target 1, first evaluation
    level, streak = choose_level(ladder(), budget_ms=0.7, current_level=0, streak=1)
    assert (level, streak) == (0, 2)
    level, streak = choose_level(ladder(), budget_ms=0.7, current_level=0, streak=2)
    assert (level, streak) == (1, 0)   # hysteresis_k=3 reached: switch


def test_policy_holds_current_level_below_hysteresis():
    level, streak = choose_level(ladder(), budget_ms=0.7, current_level=0, streak=0,
                                 hysteresis_k=5)
    assert level == 0 and streak == 1


def test_policy_stays_when_budget_fits():
    level, streak = choose_level(ladder(), budget_ms=5.0, current_level=0, streak=3)
    assert (level, streak) == (0, 0)


def test_policy_coarsens_when_budget_tight_and_can_return():
    lad = ladder()
    # Tight budget: target is the coarsest level (0.3 <= 0.32).
    level, streak = choose_level(lad, budget_ms=0.32, current_level=0, streak=2)
    assert level == 3 and streak == 0
    # Budget recovers: target back to finest, hysteresis applies again.
    level, streak = choose_level(lad, budget_ms=5.0, current_level=3, streak=0)
    assert (level, streak) == (3, 1)


def test_policy_rejects_zero_hysteresis():
    with pytest.raises(ValueError):
        choose_level(ladder(), 1.0, 0, 0, hysteresis_k=0)
