"""Track eigenmode identity across frames.

Phase 0 returns eigenvalues in ascending order. That order swaps labels when
two frequencies cross, and a near-degenerate pair can rotate inside its
subspace without a physical change of either shape. Sign is arbitrary.

Identity is the previous tracked vector. The overlap uses the same mass
matrix that produced the vectors (the M-inner product). A Hungarian match on
absolute overlap removes ordinary permutations and sign flips. Frequencies
within ``abs_tol + rel_tol * max(|f|)`` (default ``0.01 Hz`` and ``1e-3``)
are one subspace: an orthogonal Procrustes rotation aligns that block, so a
pure presentation change does not look like a new mode.

Tracked frequency is the Rayleigh quotient of the aligned vector, in hertz.
It can move continuously through a crossing. Sorted eigenvalues stay available
for comparison and are not themselves the identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def _clusters(freqs: np.ndarray, abs_tol: float, rel_tol: float) -> list[list[int]]:
    order = list(np.argsort(freqs))
    groups: list[list[int]] = []
    current = [order[0]]
    for index in order[1:]:
        previous = current[-1]
        gap = abs(float(freqs[index]) - float(freqs[previous]))
        limit = abs_tol + rel_tol * max(abs(float(freqs[index])), abs(float(freqs[previous])))
        if gap <= limit:
            current.append(index)
        else:
            groups.append(sorted(current))
            current = [index]
    groups.append(sorted(current))
    return groups


def _polar(overlap: np.ndarray) -> np.ndarray:
    """Orthogonal factor R maximizing trace(overlap @ R)."""
    left, _, right_t = np.linalg.svd(overlap, full_matrices=True)
    return right_t.T @ left.T


def _rayleigh_hz(vector, stiffness, mass) -> float:
    numerator = float(vector @ stiffness @ vector)
    denominator = float(vector @ mass @ vector)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan")
    return float(np.sqrt(max(numerator / denominator, 0.0)) / (2.0 * np.pi))


@dataclass
class ModeTracker:
    abs_tol_hz: float = 1e-2
    rel_tol: float = 1e-3
    prev_vectors: np.ndarray | None = None
    prev_freqs: np.ndarray | None = None
    frames: int = 0
    history: list[dict] = field(default_factory=list)

    def update(self, freqs, vectors, mass, stiffness) -> dict:
        freqs = np.asarray(freqs, dtype=np.float64).reshape(-1)
        vectors = np.asarray(vectors, dtype=np.float64)
        mass = np.asarray(mass, dtype=np.float64)
        stiffness = np.asarray(stiffness, dtype=np.float64)
        if vectors.ndim != 2 or vectors.shape[1] != freqs.shape[0]:
            raise ValueError("vectors must have one column per frequency")
        if self.prev_vectors is None:
            self.prev_vectors = vectors.copy()
            self.prev_freqs = freqs.copy()
            self.frames = 1
            report = self._report(
                freqs, freqs, vectors, np.ones(freqs.shape[0]),
                [], _clusters(freqs, self.abs_tol_hz, self.rel_tol), list(range(freqs.shape[0])),
            )
            self.history.append(report)
            return report
        if vectors.shape != self.prev_vectors.shape:
            raise ValueError("mode count changed; this tracker keeps a fixed identity set")
        overlap = self.prev_vectors.T @ mass @ vectors
        row, col = linear_sum_assignment(1.0 - np.abs(overlap))
        assignment = np.full(freqs.shape[0], -1, dtype=int)
        assignment[row] = col
        groups = _clusters(freqs, self.abs_tol_hz, self.rel_tol)
        aligned = np.zeros_like(vectors)
        sign_flips: list[int] = []
        degenerate: list[list[int]] = []
        for group in groups:
            prev_ids = sorted(int(np.where(assignment == new_index)[0][0]) for new_index in group)
            block = self.prev_vectors[:, prev_ids].T @ mass @ vectors[:, group]
            rotation = _polar(block)
            mixed = vectors[:, group] @ rotation
            if len(prev_ids) == 1 and float(rotation[0, 0]) < 0.0:
                sign_flips.append(prev_ids[0])
            for local, prev_id in enumerate(prev_ids):
                column_overlap = float(self.prev_vectors[:, prev_id] @ mass @ mixed[:, local])
                if column_overlap < 0.0:
                    mixed[:, local] *= -1.0
                    if prev_id not in sign_flips:
                        sign_flips.append(prev_id)
                aligned[:, prev_id] = mixed[:, local]
            if len(prev_ids) > 1:
                degenerate.append(prev_ids)
        tracked = np.array([
            _rayleigh_hz(aligned[:, index], stiffness, mass) for index in range(freqs.shape[0])
        ])
        overlaps = np.array([
            float(self.prev_vectors[:, index] @ mass @ aligned[:, index])
            for index in range(freqs.shape[0])
        ])
        self.prev_vectors = aligned
        self.prev_freqs = tracked.copy()
        self.frames += 1
        report = self._report(
            freqs, tracked, aligned, overlaps, sign_flips, groups, assignment.tolist(),
            degenerate,
        )
        self.history.append(report)
        return report

    def _report(self, sorted_freqs, tracked, vectors, overlaps, sign_flips, groups,
                assignment, degenerate=None) -> dict:
        return {
            "frame": self.frames,
            "sorted_frequencies_hz": [float(value) for value in sorted_freqs],
            "tracked_frequencies_hz": [float(value) for value in tracked],
            "overlap": [float(value) for value in overlaps],
            "sign_flips": [int(value) for value in sign_flips],
            "degenerate_tracked_ids": degenerate or [],
            "sorted_groups": groups,
            "assignment_new_column": [int(value) for value in assignment],
            "min_overlap": float(np.min(overlaps)) if len(overlaps) else 1.0,
        }
