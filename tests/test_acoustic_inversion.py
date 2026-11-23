"""Acoustic-observation inversion — glossary of record (before the code).

Recorded decisions:

- Observation: an impact WAVEFORM only. The estimator never sees the
  trajectory that produced the sound. Peeking at dynamics data would make
  the claim "identifies material by listening" vacuous.
- Acoustic forward model: candidate (mu, speed) -> impact simulation
  through the integrator (langevin_step, Coulomb term) -> velocity jump
  at contact -> pair-potential-Hessian modes -> participation weights ->
  synthesized waveform. Same object as G2; the integrator is inside the
  loop, so this is not a curve fit against a lookup table.
- Objective: mean squared error over the excitation epoch only (first
  0.2 s, recorded): with tau = 0.05 s the exponential tail is mu-silent —
  every mode decays identically regardless of participation. The mu
  signature lives in WHO rings HOW LOUD at the impact; scoring the tail
  would only add noise to a flat landscape.
- Recorded identifiability finding: jointly optimizing (mu, speed)
  against the post-impact sound alone is DEGENERATE — a continuum of
  (mu, speed) pairs reproduces the same excitation within observation
  noise. Measured live: per-start objectives all reach the noise floor at
  different (mu, speed). The estimator is therefore two-step: (1) impact
  timing is read from the waveform (first sample above 20% of peak) and
  speed comes from the protocol geometry delay = gap / (2 speed) — the
  forward model prepends the approach time as silence, making timing
  observable; (2) mu is optimized in 1D at that speed.
- E is fixed for this slice (E_fixed); joint E+mu from sound alone is not
  claimed. speed stays the declared nuisance.
- Loop reuse: run_closed_loop is observation-agnostic (recorded): its
  ``invert`` callable receives a growing prefix, which for waveform input
  is an audio window. Listening longer = knowing better is the same
  windowed semantics as the trajectory loop.
- Start grid (recorded): mu=0 is a valid boundary answer, so the grid
  always includes a near-boundary start; the 1D landscape can hold local
  minima between basins, so the loop configuration uses four starts.
- Gates: mu_true=0.5 recovered from sound within 30%; the mu=0 control
  stays under 0.1 (with the recorded observation noise of 0.002 RMS);
  determinism; the forward model counts integrator calls. Windowed
  acoustic estimates are admitted a looser 50% gate — short audio windows
  carry less excitation evidence; the record keeps the per-window values
  so the loosening is visible, not silent.
"""
import numpy as np
import pytest

from inference.inverse_params import invert_mu_acoustic
from loop.closed_loop import run_closed_loop
from queries.impact_sound import simulate_impact_sound
from occupancy_gaussian_framework import OccupancyGaussian

MU_TRUE = 0.5
SLIDE_T = 0.15
E_TRUE = 1.0e5
SPEED_TRUE = 0.2
SLIDE = 0.1
SR = 44100
DURATION = 0.6
DT = 1.0e-4


def scene(dist=0.13, scale=0.05):
    return [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=E_TRUE),
        OccupancyGaussian(pos=np.array([+dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * scale, mass=1.0, E=E_TRUE),
    ]


@pytest.fixture(scope="module")
def sound_world():
    gs = scene()
    wave, meta = simulate_impact_sound(E_TRUE, SPEED_TRUE, MU_TRUE, gs,
                                       slide=SLIDE_T, dt=DT, sr=SR,
                                       duration_s=DURATION,
                                       approach_distance=0.03)
    rng = np.random.default_rng(23)
    noisy = wave + rng.normal(0.0, 0.002, wave.shape).astype(np.float32)
    return gs, noisy, meta


def test_recovers_mu_from_sound_alone(sound_world):
    gs, observed, _ = sound_world
    report = invert_mu_acoustic(observed, gs, sr=SR, n_inits=3,
                                slide=SLIDE_T, approach_distance=0.03)
    rel = abs(report["mu_est"] - MU_TRUE) / MU_TRUE
    assert rel < 0.3, f"mu_est {report['mu_est']} rel {rel}"
    assert report["integrator_calls"] > 0


def test_mu_zero_control_from_sound():
    gs = scene()
    wave, _ = simulate_impact_sound(E_TRUE, SPEED_TRUE, 0.0, gs,
                                    slide=SLIDE_T, dt=DT, sr=SR,
                                    duration_s=DURATION,
                                    approach_distance=0.03)
    rng = np.random.default_rng(29)
    observed = wave + rng.normal(0.0, 0.002, wave.shape).astype(np.float32)
    report = invert_mu_acoustic(observed, gs, sr=SR, n_inits=3,
                                slide=SLIDE_T, approach_distance=0.03)
    assert report["mu_est"] < 0.1


def test_deterministic_from_sound(sound_world):
    gs, observed, _ = sound_world
    a = invert_mu_acoustic(observed, gs, sr=SR, n_inits=2, slide=SLIDE_T, approach_distance=0.03)
    b = invert_mu_acoustic(observed, gs, sr=SR, n_inits=2, slide=SLIDE_T, approach_distance=0.03)
    assert a["mu_est"] == b["mu_est"] and a["objective"] == b["objective"]


def test_audio_windows_through_the_loop(sound_world):
    """Listening longer is the same windowed semantics: the loop slices
    waveform prefixes, each inverted through the acoustic forward model."""
    gs, observed, _ = sound_world
    result = run_closed_loop(observed, gs, window_steps=SR // 2,
                             max_windows=3,
                             invert=lambda prefix, g: invert_mu_acoustic(
                                 prefix, g, sr=SR, n_inits=4,
                                 slide=SLIDE_T, approach_distance=0.03))
    assert result["passed"], result["failures"]
    assert 2 <= len(result["windows"]) <= 3
    assert abs(result["windows"][-1]["mu_est"] - MU_TRUE) / MU_TRUE < 0.5
    assert "mu_est" in result["windows"][-1]
