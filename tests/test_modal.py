"""Test: modal frequencies are positive and ordered."""
import numpy as np
from occupancy_gaussian_framework import (OccupancyGaussian, assemble_K_M,
                                         modal_analysis)

def test_modal_sanity():
    gs = [OccupancyGaussian(pos=np.array([0.,0.,0.]), scale=np.ones(3)*.2, mass=1., tag="a"),
          OccupancyGaussian(pos=np.array([1.,0.,0.]), scale=np.ones(3)*.2, mass=1., tag="b")]
    K, M = assemble_K_M(gs)
    freqs, _ = modal_analysis(K, M, n_modes=4)
    assert np.all(freqs >= 0), "negative frequencies"
    assert np.all(np.diff(freqs) >= -1e-9), "frequencies not ordered"
