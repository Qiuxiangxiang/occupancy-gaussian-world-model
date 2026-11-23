"""Phase 5: LOD pyramid by moment-preserving gaussian merges.

Design decisions live in PLAN.md (Phase 5) and tests/test_lod.py (glossary
of record). Summary:

- merge(g1, g2) conserves total mass, first moment, and second moment
  exactly; the merged covariance is projected to the diagonal scales Phase 0
  integrates; the off-diagonal residual is measured and reported.
- The merge tree is deterministic: KD-style median split along the longest
  axis, no RNG.
- The merge tree respects interaction topology: merges happen only inside
  1-sigma connectivity components (contact_between at threshold 0). Merging
  across a gap would teleport mass into the contact zone and manufacture
  interactions that do not exist at the finer level; that is forbidden.
- Disconnected elements persist: a level that cannot merge any component
  further terminates the pyramid early instead of looping. Rigid reduction
  (P1-2) would change what is mergeable; until then, LOD coarse-grains
  only what is actually connected.
- E/nu/rho are mass-weighted means; merged mass is the sum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from occupancy_gaussian_framework import ContactPair, OccupancyGaussian, contact_between


@dataclass
class MergeReport:
    mass_relative_error: float
    com_drift_m: float
    offdiag_residual_m2: float   # sum of squared off-diagonal cov entries


def _second_moment(g: OccupancyGaussian) -> np.ndarray:
    return g.cov + np.outer(g.pos, g.pos)


def merge_pair(g1: OccupancyGaussian, g2: OccupancyGaussian) -> OccupancyGaussian:
    """Moment-preserving merge of two gaussians.

    Conserves mass, first moment, and second moment; the merged covariance
    is projected onto world axes (diagonal), which is what the Phase 0
    integrator supports. The off-diagonal residual is the caller's to
    report via merge_diagnostics.
    """
    m1, m2 = float(g1.mass), float(g2.mass)
    total = m1 + m2
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("merged mass must be positive and finite")
    w1, w2 = m1 / total, m2 / total
    pos = w1 * g1.pos + w2 * g2.pos
    second = w1 * _second_moment(g1) + w2 * _second_moment(g2)
    cov = second - np.outer(pos, pos)
    diag = np.diag(cov).copy()
    if np.any(diag < 0.0):
        # Numerical roundoff can push a near-zero variance negative.
        floor = 1e-12
        if np.any(diag < -1e-9):
            raise ValueError(f"merged covariance lost positive semidefiniteness: {diag}")
        diag = np.maximum(diag, floor)
    scale = np.sqrt(diag)
    merged = OccupancyGaussian(
        pos=pos, scale=scale, mass=total,
        tag=f"({g1.tag}+{g2.tag})" if g1.tag != g2.tag else g1.tag,
        E=w1 * g1.E + w2 * g2.E,
        nu=w1 * g1.nu + w2 * g2.nu,
        rho=w1 * g1.rho + w2 * g2.rho,
    )
    return merged


def merge_diagnostics(children: list[OccupancyGaussian],
                      merged: OccupancyGaussian) -> MergeReport:
    mass_before = sum(float(g.mass) for g in children)
    mass_after = float(merged.mass)
    com_before = sum(float(g.mass) * g.pos for g in children) / mass_before
    offdiag = 0.0
    # Reconstruct the pre-projection covariance to measure what the diagonal
    # projection dropped.
    total = mass_before
    second = sum(float(g.mass) * _second_moment(g) for g in children) / total
    cov = second - np.outer(merged.pos, merged.pos)
    mask = ~np.eye(3, dtype=bool)
    offdiag = float(np.sum(cov[mask] ** 2))
    return MergeReport(
        mass_relative_error=abs(mass_after - mass_before) / mass_before,
        com_drift_m=float(np.linalg.norm(merged.pos - com_before)),
        offdiag_residual_m2=offdiag,
    )


def _components(gaussians: list[OccupancyGaussian]) -> list[list[int]]:
    """Connected components under the 1-sigma overlap predicate, union-find,
    deterministic order."""
    n = len(gaussians)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(n):
        for j in range(i + 1, n):
            if contact_between(gaussians, i, j, threshold=0.0) is not None:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for _, v in sorted(groups.items())]


def build_lod_pyramid(gaussians: list[OccupancyGaussian],
                      min_count: int = 2,
                      respect_connectivity: bool = True,
                      collect_reports: bool = False):
    """Returns levels, or (levels, reports) when collect_reports is set;
    reports carry one MergeReport per merge actually performed."""
    """Deterministic LOD pyramid: level 0 is the input, each next level
    halves the count by moment-preserving merges along a KD-style tree.

    With ``respect_connectivity`` (default), the KD pairing runs inside each
    1-sigma connectivity component separately; components never merge with
    each other, so inter-body gaps survive every level.
    """
    if min_count < 1:
        raise ValueError("min_count must be >= 1")
    levels = [list(gaussians)]
    reports: list[MergeReport] = []
    current = list(gaussians)
    while len(current) > min_count:
        n = len(current)
        blocks = _components(current) if respect_connectivity else [list(range(n))]
        next_level = []
        for block in blocks:
            order = _kd_order([current[i] for i in block])
            order = [block[k] for k in order]
            pairs = [(order[i], order[i + 1]) for i in range(0, len(order) - 1, 2)]
            used = {i for pair in pairs for i in pair}
            for a, b in pairs:
                merged = merge_pair(current[a], current[b])
                if collect_reports:
                    reports.append(merge_diagnostics([current[a], current[b]], merged))
                next_level.append(merged)
            for i in order:
                if i not in used:
                    next_level.append(current[i])
        if len(next_level) == len(current):
            break  # nothing could merge; disconnected elements persist
        current = next_level
        levels.append(current)
    if collect_reports:
        return levels, reports
    return levels


def _kd_order(gaussians: list[OccupancyGaussian]) -> list[int]:
    """Deterministic merge order: median split along the longest axis,
    applied recursively, pairing adjacent leaves of the resulting order."""
    indices = list(range(len(gaussians)))

    def rec(idxs: list[int]) -> list[int]:
        if len(idxs) <= 2:
            return idxs
        pts = np.stack([gaussians[i].pos for i in idxs])
        spans = pts.max(axis=0) - pts.min(axis=0)
        axis = int(np.argmax(spans))
        order = sorted(idxs, key=lambda i: (gaussians[i].pos[axis], i))
        mid = len(order) // 2
        return rec(order[:mid]) + rec(order[mid:])

    return rec(indices)


# --- adaptive policy (pure function) ----------------------------------------

@dataclass
class Ladder:
    """Frame-cost ladder: counts[i] gaussians cost costs[i] ms per frame."""
    counts: list[int]
    costs_ms: list[float]

    def cost_for(self, count: int) -> float:
        for c, cost in zip(self.counts, self.costs_ms):
            if c <= count:
                return cost
        return self.costs_ms[-1]


def choose_level(ladder: Ladder, budget_ms: float, current_level: int,
                 streak: int, *, hysteresis_k: int = 3) -> tuple[int, int]:
    """Budget-driven level choice with hysteresis.

    Returns (new_level, new_streak). Levels are indexed 0 = finest. The
    target is the finest level whose cost fits the budget. A switch only
    happens after the target differs from the current level for
    `hysteresis_k` consecutive evaluations (including this one); until then
    the current level holds and the streak accumulates. Same target resets
    nothing; reaching the threshold switches and resets the streak.
    """
    if hysteresis_k < 1:
        raise ValueError("hysteresis_k must be >= 1")
    target = 0
    for level, cost in enumerate(ladder.costs_ms):
        if cost <= budget_ms:
            target = level
            break
    else:
        target = len(ladder.costs_ms) - 1
    if target == current_level:
        return current_level, 0
    new_streak = streak + 1
    if new_streak >= hysteresis_k:
        return target, 0
    return current_level, new_streak
