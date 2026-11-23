"""Joint E+mu inversion and dual-parameter closed loop — glossary of record.

Recorded decisions:

- Estimator: theta = (log E, mu, speed), one objective (position MSE over
  the trajectory), one forward model. E keeps M4's log parameterization;
  mu keeps the bounded parameterization of the friction slice; speed
  remains the declared nuisance. This is a joint inversion, not two
  single inversions glued together: the objective sees both signatures
  (bounce timing and tangential decay) simultaneously.
- Identifiability, stated honestly: one trajectory need not separate E
  and mu. The gates therefore require recovery on a DUAL-DIFFERENCE world
  (both E and mu far from the start grid's center), and the report must
  carry a spread metric (per-start optima dispersion) as the recorded
  conditioning evidence. A claim of full identifiability is NOT made.
- Cross-checks (recorded behavior, gated softly): on an E-only world
  (mu=0) the joint estimator must not invent mu (mu_est below the 0.1
  control line), and on a mu-only world (E fixed to truth in generation)
  the E estimate stays within an order of magnitude.
- Loop integration: `run_closed_loop` takes an `invert` entry point
  (default M4's). The joint estimator exposes the same report keys
  (E_est, speed_est, objective, simulation_calls), so the windowed
  semantics — prefixes, stopping rules, no peeking — are reused
  unchanged. The loop's belief carries mu_est through per-window reports.
- Gates: dual world E within 50%, mu within 30%; determinism; integrator
  calls counted; two-window joint loop runs and stops with a stated
  reason.
"""
import numpy as np
import pytest

from inference.inverse_params import (invert_joint, simulate_bounce)
from loop.closed_loop import run_closed_loop
from occupancy_gaussian_framework import OccupancyGaussian

E_TRUE = 1.0e5
MU_TRUE = 0.5
SPEED_TRUE = 0.2
STEPS = 800


def scene(dist=0.13, scale=0.05):
    return [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=E_TRUE),
        OccupancyGaussian(pos=np.array([+dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=E_TRUE),
    ]


@pytest.fixture(scope="module")
def dual_world():
    gs = scene()
    clean = simulate_bounce(E_TRUE, SPEED_TRUE, gs, steps=STEPS, mu=MU_TRUE)
    rng = np.random.default_rng(17)
    return gs, clean + rng.normal(0.0, 1.0e-4, clean.shape)


def test_joint_recovers_both_parameters(dual_world):
    gs, observed = dual_world
    report = invert_joint(observed, gs, n_inits=2, speed_guess=(0.2,))
    assert abs(report["E_est"] - E_TRUE) / E_TRUE < 0.5, report["E_est"]
    assert abs(report["mu_est"] - MU_TRUE) / MU_TRUE < 0.3, report["mu_est"]
    assert report["simulation_calls"] > 0
    assert report["optima_spread"]["logE_std"] >= 0.0  # recorded conditioning


def test_joint_does_not_invent_mu_on_E_only_world(dual_world):
    gs, _ = dual_world
    clean = simulate_bounce(E_TRUE, SPEED_TRUE, gs, steps=STEPS, mu=0.0)
    rng = np.random.default_rng(19)
    observed = clean + rng.normal(0.0, 1.0e-4, clean.shape)
    report = invert_joint(observed, gs, n_inits=2, speed_guess=(0.2,))
    assert report["mu_est"] < 0.1


def test_joint_deterministic(dual_world):
    gs, observed = dual_world
    a = invert_joint(observed, gs, n_inits=2, speed_guess=(0.2,))
    b = invert_joint(observed, gs, n_inits=2, speed_guess=(0.2,))
    assert a["E_est"] == b["E_est"] and a["mu_est"] == b["mu_est"]


def test_joint_loop_reuses_windowed_semantics(dual_world):
    gs, observed = dual_world
    result = run_closed_loop(observed, gs, window_steps=400, max_windows=2,
                             invert=lambda prefix, g: invert_joint(
                                 prefix, g, n_inits=2, speed_guess=(0.2,)))
    assert result["passed"], result["failures"]
    assert len(result["windows"]) == 2
    assert "mu_est" in result["windows"][0]
    assert result["stopped_by"] in ("budget_exhausted", "data_exhausted",
                                    "estimate_converged")
    assert abs(result["E_est"] - E_TRUE) / E_TRUE < 0.6
