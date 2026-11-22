"""Perception frontend: 3DGS reconstruction -> occupancy gaussians (Phase 3 stub)."""
from __future__ import annotations
import numpy as np
from occupancy_gaussian_framework import OccupancyGaussian


def load_3dgs(path: str):
    """Load a 3DGS checkpoint/ply. Stub: returns None."""
    raise NotImplementedError("Wire gsplat loader here in Phase 3")


def to_occupancy(gs_model, surface_samples: int = 1000,
                 scale: float = 0.05) -> list[OccupancyGaussian]:
    """Sample the reconstructed surface and place occupancy gaussians.
    Positions locked to surface samples; scales uniform; masses from
    local density estimate. Phase-3 work: thin-structure recovery."""
    raise NotImplementedError("Implement surface sampling + mass assignment")
