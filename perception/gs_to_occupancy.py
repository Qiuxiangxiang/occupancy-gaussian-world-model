"""Perception frontend.

Phase 3 still owns learned reconstruction (gsplat) and surface resampling.
An already written Cross3DGS point cloud is imported by ``compiler.scene_compiler``.
This module does not convert occupancy gaussians back into a point cloud for
rendering; a frame projects the occupancy gaussians directly.
"""
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
