"""Modal / acoustic query interface (Phase 2 stub + minimal working version)."""
from __future__ import annotations
import numpy as np
from occupancy_gaussian_framework import OccupancyGaussian, assemble_K_M, modal_analysis
import config as C


def query_frequencies(gaussians: list[OccupancyGaussian], n_modes: int = C.N_MODES):
    """Return list of (frequency_Hz, mode_shape_vector) for the object set."""
    K, M = assemble_K_M(gaussians)
    freqs, V = modal_analysis(K, M, n_modes=n_modes)
    return [(float(f), V[:, i]) for i, f in enumerate(freqs)]


def query_sound_spectrum(gaussians: list[OccupancyGaussian],
                         strike_gaussian: int = 0,
                         duration: float = 0.5,
                         sr: int = 44100) -> np.ndarray:
    """Synthesize a decaying modal superposition struck at strike_gaussian.
    Phase-0 placeholder: equal-amplitude modes, exponential decay."""
    modes = query_frequencies(gaussians)
    n = int(duration * sr)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for f, shape in modes[: C.N_MODES]:
        if f < 1e-6:
            continue
        amp = float(abs(shape[3 * strike_gaussian:3 * strike_gaussian + 3]).mean())
        decay = np.exp(-2 * np.pi * f * 0.02 * t)  # 2% damping per cycle-ish
        sig += amp * decay * np.sin(2 * np.pi * f * t)
    # normalize
    sig = sig / (np.max(np.abs(sig)) + 1e-12)
    return sig
