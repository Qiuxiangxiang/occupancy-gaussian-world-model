"""Scene: rigid cup + hinged lamp arm (Phase 1)."""
from __future__ import annotations
import numpy as np
from occupancy_gaussian_framework import OccupancyGaussian, State, langevin_step, contact_pairs, penalty_force
import config as C


def build():
    """Cup = rigid cluster of gaussians; lamp arm = hinge constraint (Phase 1)."""
    cup = [OccupancyGaussian(pos=np.array([0.0, 0.0, 0.0]) + np.random.randn(3) * 0.05,
                             scale=np.ones(3) * 0.08, mass=0.2, tag="cup") for _ in range(8)]
    lamp = [OccupancyGaussian(pos=np.array([0.6, 0.4, 0.0]) + np.array([0.0, i * 0.05, 0.0]),
                             scale=np.ones(3) * 0.05, mass=0.1, tag="lamp") for i in range(6)]
    return cup + lamp


if __name__ == "__main__":
    print("cup_and_lamp scene: Phase 1 — hinge constraint + rigid reduction not yet wired")
    gs = build()
    print(f"built {len(gs)} gaussians: {sum(1 for g in gs if g.tag=='cup')} cup, {sum(1 for g in gs if g.tag=='lamp')} lamp")
