"""G2 gate: Phase 2 acoustics acceptance.

Glossary of record: tests/test_acoustics.py. The gate simulates a real
impact, captures the velocity jump at contact-on, computes the impacting
object's acoustic modes from the pair-potential Hessian, synthesizes the
impact sound, and checks the spectral peak against the excited mode. A WAV
artifact is written. FEM reference: open and recorded (no FEM in this
repository; the sqrt(2) structural anchor to the Phase 0 reporting layer
is the recorded comparison instead).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "g2_report.json"
WAV = ROOT / "artifacts" / "g2_impact.wav"

DT = 1.0e-4


def run_g2(artifact: Path = ARTIFACT, wav: Path = WAV) -> dict:
    from queries.modal_query import local_modes, synthesize_impact
    from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                              contact_pairs, langevin_step,
                                              penalty_force)
    gaussians = [
        OccupancyGaussian(pos=np.array([-0.06, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
        OccupancyGaussian(pos=np.array([0.06, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=2.36, E=1.0e5),
    ]
    masses = np.array([2.36, 2.36])
    state = State(pos=np.stack([g.pos for g in gaussians]),
                  vel=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]))
    pre_vel = None
    post_vel = None
    for step in range(int(round(0.3 / DT))):
        for i in range(2):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        force = np.zeros((2, 3))
        for pair in pairs:
            fi, fj = penalty_force(pair, gaussians, mu=0.0)
            force[pair.i] += fi
            force[pair.j] += fj
        if pairs and pre_vel is None:
            pre_vel = state.vel.copy()
        if pre_vel is not None and post_vel is None and not pairs:
            post_vel = state.vel.copy()
            break
        state = langevin_step(state, force, DT, masses, T=0.0, gamma=0.0)
    if post_vel is None:
        post_vel = state.vel.copy()

    freqs, shapes, negative = local_modes(gaussians, [0, 1], n_modes=6)
    dv = (post_vel - pre_vel).reshape(-1)
    signal, meta = synthesize_impact(freqs, shapes, masses, dv,
                                     duration_s=1.0, sr=44100)
    meta["write_wav"](str(wav))
    spectrum = np.abs(np.fft.rfft(signal))
    axis = np.fft.rfftfreq(len(signal), d=1 / 44100)
    band = (axis >= 20) & (axis <= 20000)
    peak = float(axis[band][np.argmax(spectrum[band])])
    weights = meta["participation"]
    excited = meta["frequencies_hz"][int(np.argmax(weights))]

    failures = []
    if abs(peak - excited) / excited >= 0.02:
        failures.append(f"spectral peak {peak} Hz off excited mode {excited} Hz")
    if not wav.is_file():
        failures.append("WAV artifact missing")
    report = {"gate": "G2", "phase": "P2", "passed": not failures,
              "failures": failures,
              "wav_artifact": str(wav),
              "impact": {"pre_rel_speed": float(np.linalg.norm(pre_vel[1] - pre_vel[0])),
                         "post_rel_speed": float(np.linalg.norm(post_vel[1] - post_vel[0]))},
              "modes_hz": [float(f) for f in freqs],
              "negative_clamped_count": negative,
              "excited_mode_hz": excited,
              "spectral_peak_hz": peak,
              "participation": weights,
              "tau_s": meta["tau_s"],
              "fem_reference": ("OPEN: no FEM exists in this repository; the "
                                "recorded comparison is the sqrt(2) structural "
                                "anchor to the Phase 0 reporting layer "
                                "(tests/test_acoustics.py)")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[G2] passed={report['passed']} excited={excited:.2f} Hz "
          f"peak={peak:.2f} Hz wav={wav.name} -> {artifact}")
    return report


if __name__ == "__main__":
    outcome = run_g2()
    raise SystemExit(0 if outcome["passed"] else 1)
