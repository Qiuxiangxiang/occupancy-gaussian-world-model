"""M4: 1D Young's modulus inversion through the real integrator."""
import numpy as np
import pytest

from inference.inverse_params import invert_young_modulus, simulate_bounce
from occupancy_gaussian_framework import OccupancyGaussian, langevin_step


def scene(dist=0.13, scale=0.05):
    return [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=1.0e5),
        OccupancyGaussian(pos=np.array([+dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=1.0e5),
    ]


STEPS = 800
E_TRUE = 1.0e5
SPEED_TRUE = 0.2


def observed(seed=None, sigma=1.0e-4):
    gs = scene()
    clean = simulate_bounce(E_TRUE, SPEED_TRUE, gs, steps=STEPS)
    if seed is None:
        return clean
    rng = np.random.default_rng(seed)
    return clean + rng.normal(0.0, sigma, clean.shape)


def test_forward_model_bounces_and_uses_integrator(monkeypatch):
    calls = {"n": 0}
    original = langevin_step
    def counting(state, forces, dt, masses, **kwargs):
        calls["n"] += 1
        return original(state, forces, dt, masses, **kwargs)
    monkeypatch.setattr("inference.inverse_params.langevin_step", counting)
    traj = simulate_bounce(E_TRUE, SPEED_TRUE, scene(), steps=STEPS)
    assert calls["n"] == STEPS
    # Contact happened: minimum center distance well below the start.
    gap = np.linalg.norm(traj[:, 1] - traj[:, 0], axis=1)
    assert gap.min() < 0.11


@pytest.fixture(scope="module")
def main_report():
    data = observed(seed=7)
    return invert_young_modulus(data, scene(), noise_seed=7,
                                n_E_inits=3, speed_guess=(0.1, 0.3))


def test_inversion_recovers_modulus_order_of_magnitude(main_report):
    report = main_report
    rel = abs(report["E_est"] - E_TRUE) / E_TRUE
    assert rel < 0.5, f"E_est {report['E_est']} vs E_true {E_TRUE} (rel {rel})"
    # Order-of-magnitude claim is stronger than the bound above; check it.
    assert 0.5 < report["E_est"] / E_TRUE < 2.0
    # Nuisance speed is in the right range.
    assert 0.5 < report["speed_est"] / SPEED_TRUE < 2.0
    assert report["simulation_calls"] > 0


def test_inversion_is_deterministic_for_fixed_observation():
    data = observed(seed=11)
    a = invert_young_modulus(data, scene(), noise_seed=11)
    b = invert_young_modulus(data, scene(), noise_seed=11)
    assert a["E_est"] == b["E_est"] and a["objective"] == b["objective"]


def test_parameterization_respects_bounds():
    data = observed(seed=3)
    report = invert_young_modulus(data, scene(), E_bounds=(3e4, 3e5), noise_seed=3)
    assert 3e4 <= report["E_est"] <= 3e5
    for start in report["starts"]:
        assert 3e4 <= start["E_est"] <= 3e5


def test_noiseless_recovery_is_tighter_than_noisy():
    clean = observed()
    rng = np.random.default_rng(5)
    noisy = clean + rng.normal(0.0, 1.0e-4, clean.shape)
    fit_clean = invert_young_modulus(clean, scene(), n_E_inits=2, speed_guess=(0.2,))
    fit_noisy = invert_young_modulus(noisy, scene(), n_E_inits=2, speed_guess=(0.2,))
    err_clean = abs(fit_clean["E_est"] - E_TRUE) / E_TRUE
    err_noisy = abs(fit_noisy["E_est"] - E_TRUE) / E_TRUE
    assert err_clean <= err_noisy + 0.05
