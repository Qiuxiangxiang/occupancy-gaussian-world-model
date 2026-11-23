"""Friction inversion slice — glossary of record (before the code).

Recorded decisions:

- Re-scope: the Phase 4 stub promised "friction from acoustic spectrum".
  Spectrum matching is blocked on Phase 2 acoustics, which is not built.
  This slice inverts the friction coefficient MU from a TANGENTIAL
  SLIDING COLLISION TRAJECTORY through the same forward model as M4
  (`simulate_bounce` with mu passed to `penalty_force`) — never a curve
  fit that skips the integrator. The acoustic route stays open and is
  recorded as such.
- Parameterization: mu is bounded directly (clip to mu_bounds, interior
  epsilon floor 1e-4); speed is the declared nuisance, estimated jointly.
  theta = (mu, speed); E is FIXED at the known value for this slice —
  joint E+mu identifiability from one trajectory is not claimed.
- Objective: mean squared error over the full trajectory (positions of
  both bodies), exactly the M4 objective; mu's signature is the
  tangential motion decay, which the position MSE sees.
- Initializations: several (mu_init x speed_init) starts, best objective
  wins; deterministic given the observation (no noise is added inside the
  estimator).
- Gates: recovery of mu_true=0.5 within 30% relative error; the mu=0
  control recovers mu < 0.05; determinism under a fixed observation; the
  objective really calls the integrator (simulation_calls > 0).
"""
import numpy as np
import pytest

from inference.inverse_params import invert_friction, simulate_bounce
from occupancy_gaussian_framework import OccupancyGaussian

MU_TRUE = 0.5
SPEED_TRUE = 0.2
STEPS = 800


def scene(dist=0.13, scale=0.05):
    return [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=1.0e5),
        OccupancyGaussian(pos=np.array([+dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=1.0e5),
    ]


@pytest.fixture(scope="module")
def world():
    gs = scene()
    clean = simulate_bounce(1.0e5, SPEED_TRUE, gs, steps=STEPS, mu=MU_TRUE)
    rng = np.random.default_rng(11)
    return gs, clean + rng.normal(0.0, 1.0e-4, clean.shape)


def test_recovers_mu(world):
    gs, observed = world
    report = invert_friction(observed, gs, n_mu_inits=3, speed_guess=(0.2,))
    rel = abs(report["mu_est"] - MU_TRUE) / MU_TRUE
    assert rel < 0.3, f"mu_est {report['mu_est']} rel {rel}"
    assert 0.5 < report["speed_est"] / SPEED_TRUE < 2.0
    assert report["simulation_calls"] > 0


def test_mu_zero_control(world):
    gs, _ = world
    clean = simulate_bounce(1.0e5, SPEED_TRUE, gs, steps=STEPS, mu=0.0)
    rng = np.random.default_rng(13)
    observed = clean + rng.normal(0.0, 1.0e-4, clean.shape)
    report = invert_friction(observed, gs, n_mu_inits=2, speed_guess=(0.2,))
    assert report["mu_est"] < 0.05


def test_deterministic(world):
    gs, observed = world
    a = invert_friction(observed, gs, n_mu_inits=2, speed_guess=(0.2,))
    b = invert_friction(observed, gs, n_mu_inits=2, speed_guess=(0.2,))
    assert a["mu_est"] == b["mu_est"] and a["objective"] == b["objective"]


def test_respects_bounds():
    gs = scene()
    clean = simulate_bounce(1.0e5, SPEED_TRUE, gs, steps=STEPS, mu=0.5)
    report = invert_friction(clean, gs, mu_bounds=(0.1, 0.6), n_mu_inits=2,
                             speed_guess=(0.2,))
    assert 0.1 <= report["mu_est"] <= 0.6
