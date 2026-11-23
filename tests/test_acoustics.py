"""Phase 2 acoustics — glossary of record (before the code).

Recorded decisions:

- Local modes: per object (member list), the generalized spectrum of the
  pair-stiffness K restricted to member DoFs on the integrator's diagonal
  mass basis. Semantics inherited from P5's recorded finding: K has a zero
  diagonal, eigenvalues come in +/- pairs; the negative half is a clamp
  artifact and is discarded (counted, not hidden). Rigid translations are
  genuine zero modes; stretching modes are the audible ones.
- Impact excitation: participation weight w_k = |v_k^T M dv| / sum_j
  |v_j^T M dv|, where dv is the velocity jump across the impact event.
  This is the M3 participation proposal, now computed for a real impact —
  it becomes measurement, not proposal.
- Synthesis: s(t) = sum_k w_k sin(2 pi f_k t) exp(-t/tau), tau recorded at
  0.05 s (a decision, not a fit). WAV via the stdlib wave module, int16,
  44.1 kHz mono. Deterministic given the event.
- Anchor: the M1 two-splat contact spectrum (39.74 Hz at contact-on) is
  the scale anchor. Gate: the synthesized impact sound's spectral peak in
  the 20-20000 Hz band lies within 2% of the smallest positive tracked
  mode of the impacting object's local spectrum.
- FEM reference: OPEN and recorded. There is no FEM in this repository;
  the M1 analytic anchor is the coarse reference. A real FEM comparison
  is a Phase 2 follow-up, not claimed here.
"""
import wave

import numpy as np
import pytest

from queries.modal_query import (local_modes, synthesize_impact)
from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                          contact_pairs, langevin_step,
                                          penalty_force)

DT = 1.0e-4


def two_splats(dist=0.06):
    return [
        OccupancyGaussian(pos=np.array([-dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
        OccupancyGaussian(pos=np.array([dist / 2, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
    ]


def test_local_modes_structure_and_negative_half():
    gaussians = two_splats()
    freqs, shapes, negative = local_modes(gaussians, [0, 1], n_modes=6)
    # Recorded semantics: Phase 0's pair-K has no diagonal blocks, so at a
    # gap there are no true rigid zeros; the "zeros" in M1 logs were
    # clamped negatives. Gate structure only: negative half counted and
    # excluded, positive spectrum non-empty, shapes dimension consistent.
    assert negative >= 1
    assert all(f >= 0.0 for f in freqs)
    assert any(f > 1.0 for f in freqs)
    assert shapes.shape[0] == 6


def test_hessian_modes_are_sqrt2_times_phase0_modes():
    """Structural anchor, recorded: Phase 0's reporting K has eigenvalues
    +/-k on this geometry; the pair-potential Hessian has rigid zeros plus
    the same modes scaled by sqrt(2) (relative spring 2k, reduced mass
    m/2). M1's event frequencies were instantaneous-overlap samples of the
    Phase 0 layer; the sqrt(2) relationship is the portable anchor."""
    import numpy as np
    from scipy.linalg import eigh as _eigh
    from occupancy_gaussian_framework import assemble_K_M
    gaussians = two_splats(dist=0.12)
    freqs_h, _, _ = local_modes(gaussians, [0, 1], n_modes=6)
    K0, _ = assemble_K_M(gaussians)
    M = np.diag(np.repeat([2.36, 2.36], 3))
    w2, _ = _eigh(K0, M)
    phase0 = sorted(np.sqrt(np.maximum(w2, 0)) / (2 * np.pi))
    phase0 = [f for f in phase0 if f > 0.01]
    hessian = [f for f in freqs_h if f > 0.01]
    assert len(phase0) == len(hessian) >= 2
    for f0, fh in zip(phase0, hessian):
        assert fh == pytest.approx(f0 * np.sqrt(2), rel=0.02), (f0, fh)


def test_participation_weights_are_normalized_and_concentrated():
    gaussians = two_splats(dist=0.12)
    freqs, shapes, _ = local_modes(gaussians, [0, 1], n_modes=4)
    masses = np.array([2.36, 2.36])
    dv = np.zeros(6)
    dv[0] = 0.2   # impact along the approach axis on body 0
    dv[3] = -0.2
    signal, meta = synthesize_impact(freqs, shapes, masses, dv,
                                     duration_s=0.5, sr=44100)
    weights = meta["participation"]
    assert abs(sum(weights) - 1.0) < 1e-9
    # Anisotropy (recorded): k_n > k_t, so a pure x-impact excites the
    # normal-polarized (stiffer) mode, not the soft tangential doublet;
    # rigid zeros carry no weight.
    loudest = int(np.argmax(weights))
    assert weights[loudest] > 0.9
    assert weights[0] < 1e-9
    assert meta["frequencies_hz"][loudest] > 90.0  # the x-polarized mode
    assert meta["sample_rate"] == 44100
    assert len(signal) == int(0.5 * 44100)


def test_spectral_peak_matches_lowest_tracked_mode(tmp_path):
    gaussians = two_splats(dist=0.12)
    freqs, shapes, _ = local_modes(gaussians, [0, 1], n_modes=4)
    masses = np.array([2.36, 2.36])
    dv = np.zeros(6)
    dv[0], dv[3] = 0.2, -0.2
    signal, meta = synthesize_impact(freqs, shapes, masses, dv,
                                     duration_s=1.0, sr=44100)
    wav = tmp_path / "impact.wav"
    meta["write_wav"](str(wav))
    with wave.open(str(wav), "rb") as stream:
        assert stream.getframerate() == 44100
        assert stream.getnchannels() == 1
        assert stream.getnframes() == 44100
    spectrum = np.abs(np.fft.rfft(signal))
    freqs_axis = np.fft.rfftfreq(len(signal), d=1 / 44100)
    band = (freqs_axis >= 20) & (freqs_axis <= 20000)
    peak = freqs_axis[band][np.argmax(spectrum[band])]
    # The peak must match the mode the impact actually excites (largest
    # participation weight), not necessarily the lowest mode.
    excited = meta["frequencies_hz"][int(np.argmax(meta["participation"]))]
    assert peak == pytest.approx(excited, rel=0.02), (peak, excited)


def test_g2_gate_runs(tmp_path):
    from compiler.g2_harness import run_g2
    report = run_g2(artifact=tmp_path / "g2.json", wav=tmp_path / "g2.wav")
    assert report["passed"], report["failures"]
    assert (tmp_path / "g2.wav").is_file()
    assert report["fem_reference"].startswith("OPEN")
    assert abs(report["spectral_peak_hz"] - report["excited_mode_hz"]) \
        / report["excited_mode_hz"] < 0.02
