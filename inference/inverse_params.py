"""Parameter inversion: estimate material params from observations (Phase 4 stub)."""
from __future__ import annotations
import numpy as np


def invert_young_modulus(observed_trajectory, gaussians, E_bounds=(1e4, 1e9)):
    """Optimize E from bounce trajectory. Stub."""
    raise NotImplementedError("Phase 4: gradient-free or autodiff search over E")


def invert_friction(observed_spectrum, gaussians, mu_bounds=(0.0, 1.0)):
    """Optimize friction coefficient from acoustic spectrum. Stub."""
    raise NotImplementedError("Phase 4: spectrum matching")
