"""Mode identity under permutation, sign flip, and a near-degenerate rotation."""
from __future__ import annotations

import numpy as np

from compiler.mode_tracking import ModeTracker
from occupancy_gaussian_framework import modal_analysis


def _mass_and_vectors(diag_k):
    size = len(diag_k)
    stiffness = np.diag(np.asarray(diag_k, dtype=np.float64))
    mass = np.eye(size)
    freqs, vectors = modal_analysis(stiffness, mass, n_modes=size)
    return stiffness, mass, freqs, vectors


def test_sign_flip_keeps_identity():
    stiffness, mass, freqs, vectors = _mass_and_vectors([1.0, 100.0, 400.0])
    tracker = ModeTracker()
    tracker.update(freqs, vectors, mass, stiffness)
    flipped = vectors.copy()
    flipped[:, 1] *= -1.0
    report = tracker.update(freqs, flipped, mass, stiffness)
    assert report["sign_flips"] == [1]
    assert report["min_overlap"] > 0.999
    assert np.allclose(report["tracked_frequencies_hz"], report["sorted_frequencies_hz"], rtol=1e-6, atol=1e-6)


def test_permutation_follows_the_vector_not_the_sorted_slot():
    mass = np.eye(2)
    stiffness = np.diag([1.0, 100.0])
    freqs, vectors = modal_analysis(stiffness, mass, n_modes=2)
    tracker = ModeTracker()
    tracker.update(freqs, vectors, mass, stiffness)
    # The two diagonal entries cross. eigh sorts the new pair, which swaps columns.
    crossed = np.diag([100.0, 1.0])
    new_freqs, new_vectors = modal_analysis(crossed, mass, n_modes=2)
    report = tracker.update(new_freqs, new_vectors, mass, crossed)
    tracked = np.asarray(report["tracked_frequencies_hz"])
    assert tracked[0] > tracked[1]
    assert abs(tracked[0] - 100.0 ** 0.5 / (2.0 * np.pi)) / tracked[0] < 1e-6
    assert report["min_overlap"] > 0.999


def test_near_degenerate_rotation_is_not_a_new_mode():
    stiffness, mass, freqs, vectors = _mass_and_vectors([10.0, (np.sqrt(10.0) + 0.01) ** 2])
    # Rebuild a near pair with a known subspace: two close frequencies.
    values = np.array([4.0 * np.pi ** 2 * 5.0 ** 2, 4.0 * np.pi ** 2 * 5.001 ** 2])
    stiffness = np.diag(values)
    mass = np.eye(2)
    freqs, vectors = modal_analysis(stiffness, mass, n_modes=2)
    tracker = ModeTracker()
    first = tracker.update(freqs, vectors, mass, stiffness)
    assert len(first["sorted_groups"]) == 1 and len(first["sorted_groups"][0]) == 2
    theta = 0.4
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    rotated = vectors @ rotation
    report = tracker.update(freqs, rotated, mass, stiffness)
    assert report["degenerate_tracked_ids"] == [[0, 1]]
    assert report["min_overlap"] > 0.99
    assert max(
        abs(a - b) / max(abs(b), 1.0)
        for a, b in zip(report["tracked_frequencies_hz"], first["tracked_frequencies_hz"])
    ) < 1e-3


def main():
    test_sign_flip_keeps_identity()
    test_permutation_follows_the_vector_not_the_sorted_slot()
    test_near_degenerate_rotation_is_not_a_new_mode()
    print("test_mode_tracking ok")


if __name__ == "__main__":
    main()
