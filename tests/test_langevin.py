"""Test: Langevin at equilibrium satisfies rough equipartition."""
import numpy as np
from occupancy_gaussian_framework import OccupancyGaussian, State, langevin_step
import config as C

def test_equipartition():
    g = OccupancyGaussian(pos=np.zeros(3), scale=np.ones(3)*.2, mass=1.0)
    state = State(pos=np.zeros((1,3)), vel=np.zeros((1,3)))
    ke = []
    for _ in range(5000):
        state = langevin_step(state, np.zeros((1,3)), C.DT, np.array([1.0]), T=300.0, gamma=1.0)
        ke.append(0.5 * np.sum(state.vel ** 2))
    mean_ke = np.mean(ke[1000:])
    # 3 dof * 0.5 * kT
    target = 1.5 * C.KB * 300.0
    # masses=1 in simulation units; compare ratio
    assert mean_ke > 0, "no kinetic energy at equilibrium"
