"""Phase 4 closed loop — glossary of record (before the code).

Recorded decisions:

- Loop structure: predict-observe-update. The loop receives ONE
  pre-simulated noisy trajectory (the "world"), generated with E_true at a
  fixed seed. It sees only a growing prefix: window k is the first
  k*window_steps samples. Each window is inverted through the real forward
  model (`invert_young_modulus`, M4); the estimate sequence is the belief
  trajectory. No window ever sees the future — peeking would make the
  convergence claim vacuous.
- Stopping rule (recorded): the loop stops when the relative change of the
  E estimate between successive windows falls below 2%, or when the window
  budget is exhausted. Stopping is part of the loop's output, not an
  afterthought: a closed loop must decide when it knows enough.
- Gates: (1) the final estimate error is no worse than 2x the M4
  full-trajectory single-shot error — less data cannot honestly beat more;
  (2) the loop uses strictly less observation than the full trajectory in
  at least one run of the convergence configuration (early stopping is
  real, not decorative); (3) the estimate sequence is deterministic under
  a fixed observation seed; (4) every inversion calls the real integrator
  (pinned by counting simulation calls, as in M4).
- Nuisance: the initial speed is estimated jointly at every window, as in
  M4; the loop carries no other hidden state.
"""
import numpy as np
import pytest

from compiler.g4_harness import run_g4
from inference.inverse_params import invert_young_modulus, simulate_bounce
from loop.closed_loop import run_closed_loop
from occupancy_gaussian_framework import OccupancyGaussian

E_TRUE = 1.0e5
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
def world():
    gs = scene()
    clean = simulate_bounce(E_TRUE, SPEED_TRUE, gs, steps=STEPS)
    rng = np.random.default_rng(7)
    return gs, clean + rng.normal(0.0, 1.0e-4, clean.shape)


@pytest.fixture(scope="module")
def loop_result(world):
    gs, observed = world
    return run_closed_loop(observed, gs, window_steps=150, max_windows=8,
                           n_E_inits=2, speed_guess=(0.2,))


def test_converges_and_stops_early(loop_result):
    result = loop_result
    assert result["passed"], result["failures"]
    estimates = [w["E_est"] for w in result["windows"]]
    assert estimates[-1] == pytest.approx(E_TRUE, rel=0.5)
    assert result["windows_used"] < result["windows_available"]
    assert result["stopped_by"] in ("estimate_converged", "data_exhausted")


def test_not_worse_than_full_trajectory_shot(world, loop_result):
    gs, observed = world
    full = invert_young_modulus(observed, gs, n_E_inits=2, speed_guess=(0.2,))
    loop_rel = abs(loop_result["E_est"] - E_TRUE) / E_TRUE
    full_rel = abs(full["E_est"] - E_TRUE) / E_TRUE
    assert loop_rel <= 2.0 * max(full_rel, 1e-6)


def test_deterministic_under_fixed_seed(world):
    gs, observed = world
    a = run_closed_loop(observed, gs, window_steps=150, max_windows=4,
                        n_E_inits=2, speed_guess=(0.2,))
    b = run_closed_loop(observed, gs, window_steps=150, max_windows=4,
                        n_E_inits=2, speed_guess=(0.2,))
    assert a["E_est"] == b["E_est"]
    assert [w["E_est"] for w in a["windows"]] == [w["E_est"] for w in b["windows"]]


def test_every_window_calls_the_integrator(loop_result):
    for window in loop_result["windows"]:
        assert window["simulation_calls"] > 0


def test_no_peeking(world, loop_result):
    """The windows' inversion inputs must be prefixes: re-inverting the
    reported prefix length from the full observation must reproduce the
    window estimate exactly."""
    gs, observed = world
    first = loop_result["windows"][0]
    prefix = observed[:first["steps"]]
    replay = invert_young_modulus(prefix, gs, n_E_inits=2, speed_guess=(0.2,))
    assert replay["E_est"] == first["E_est"]


def test_g4_gate_runs(tmp_path):
    report = run_g4(artifact=tmp_path / "g4_report.json")
    assert report["passed"], report["failures"]
    assert report["relative_error"] < 0.5
    assert report["stopped_by"] in ("estimate_converged", "data_exhausted")
    assert (tmp_path / "g4_report.json").is_file()
