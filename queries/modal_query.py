"""Modal / acoustic query interface (Phase 2).

Glossary of record: tests/test_acoustics.py. ``local_modes`` computes the
per-object generalized spectrum (pair-stiffness K on member DoFs,
diagonal mass basis; negative eigenvalue half discarded and counted —
P5's recorded semantics). ``synthesize_impact`` turns an impact velocity
jump into sound via participation weights w_k = |v_k^T M dv| (the M3
proposal, now measurement), synthesized with a recorded 0.05 s decay.
"""
from __future__ import annotations

import wave as _wave
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from occupancy_gaussian_framework import OccupancyGaussian, assemble_K_M

TAU_S = 0.05  # recorded synthesis decay, a decision not a fit


def local_modes(gaussians: list[OccupancyGaussian], members, n_modes: int = 6):
    """Per-object generalized spectrum on the diagonal mass basis.

    Returns (frequencies_ascending Hz, mode shapes on member DoFs,
    negative_eigenvalue_count). Zero eigenvalues are rigid modes; the
    negative half of K's +/- spectrum is discarded and counted.
    """
    members = [int(i) for i in members]
    # Acoustic modes use the pair-potential Hessian, NOT Phase 0's
    # reporting K: the Phase 0 layer assembles only off-diagonal +Kij
    # blocks, which is not the Hessian of 1/2*(x_i-x_j)^T Kij (x_i-x_j)
    # (recorded finding: its spectrum is +/- symmetric, relative-motion
    # modes land in the discarded negative half, and "zero modes" in M1
    # logs were clamped negatives). M1's numbers remain the M1 record;
    # acoustics needs the Hessian that the dynamics actually integrates.
    from occupancy_gaussian_framework import pairwise_stiffness
    dim = 3 * len(members)
    K = np.zeros((dim, dim))
    for a in range(len(members)):
        for b in range(a + 1, len(members)):
            i, j = members[a], members[b]
            Kij = pairwise_stiffness(gaussians[i], gaussians[j])
            ra, rb = slice(3 * a, 3 * a + 3), slice(3 * b, 3 * b + 3)
            K[ra, ra] += Kij
            K[rb, rb] += Kij
            K[ra, rb] -= Kij
            K[rb, ra] -= Kij
    M = np.zeros_like(K)
    for row, i in enumerate(members):
        slot = slice(3 * row, 3 * row + 3)
        M[slot, slot] = gaussians[i].mass * np.eye(3)
    count = min(int(n_modes), dim)
    w2, V = eigh(K, M)
    negative_mask = w2 < -1e-12
    negative = int(np.sum(negative_mask))
    # The pair-potential Hessian is symmetric PSD up to numerics: true
    # rigid translations sit at zero, stretching modes positive. Any tiny
    # negative eigenvalue is numerical noise and is clamped and counted.
    keep = ~negative_mask
    freqs = np.sqrt(np.maximum(w2[keep], 0.0)) / (2 * np.pi)
    V = V[:, keep]
    order = np.argsort(freqs)
    return freqs[order][:count], V[:, order][:, :count], negative


def synthesize_impact(frequencies_hz, mode_shapes, masses, dv,
                      *, duration_s: float = 1.0, sr: int = 44100,
                      tau_s: float = TAU_S):
    """Synthesize an impact sound from modal data.

    ``dv`` is the velocity jump stacked over member DoFs (3 per member,
    matching ``mode_shapes`` rows). Participation weights are
    w_k = |v_k^T M dv| / sum_j |v_j^T M dv|. Returns (signal float32,
    metadata dict; metadata['write_wav'](path) writes int16 mono WAV).
    """
    freqs = np.asarray(frequencies_hz, dtype=np.float64)
    V = np.asarray(mode_shapes, dtype=np.float64)
    dv = np.asarray(dv, dtype=np.float64)
    mass_diag = np.repeat(np.asarray(masses, dtype=np.float64), 3)
    Mdv = mass_diag * dv
    raw = np.abs(V.T @ Mdv)
    total = raw.sum()
    weights = raw / total if total > 0 else np.zeros_like(raw)
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    signal = np.zeros(n)
    for f, w in zip(freqs, weights):
        if f < 1e-9 or w <= 0:
            continue
        signal += w * np.sin(2 * np.pi * f * t) * np.exp(-t / tau_s)
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    signal = signal.astype(np.float32)

    def write_wav(path):
        pcm = (signal * 32767).astype(np.int16)
        with _wave.open(path, "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(sr)
            stream.writeframes(pcm.tobytes())

    return signal, {"sample_rate": sr, "duration_s": duration_s,
                    "participation": [float(w) for w in weights],
                    "frequencies_hz": [float(f) for f in freqs],
                    "tau_s": tau_s, "write_wav": write_wav,
                    "tau_note": "recorded decision, not a fit"}
