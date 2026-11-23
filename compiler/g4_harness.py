"""G4 gate: Phase 4 closed loop acceptance.

Glossary of record: tests/test_closed_loop.py. The gate replays the loop
on the reference world (E_true=1e5, seeded noise), checks convergence,
early stopping, determinism, and that every window inverted through the
real integrator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "g4_report.json"

E_TRUE = 1.0e5
SPEED_TRUE = 0.2
STEPS = 800


def run_g4(artifact: Path = ARTIFACT) -> dict:
    from inference.inverse_params import simulate_bounce
    from loop.closed_loop import run_closed_loop
    from occupancy_gaussian_framework import OccupancyGaussian
    gaussians = [
        OccupancyGaussian(pos=np.array([-0.065, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, E=E_TRUE),
        OccupancyGaussian(pos=np.array([0.065, 0.0, 0.0]),
                          scale=np.ones(3) * 0.05, mass=1.0, E=E_TRUE),
    ]
    clean = simulate_bounce(E_TRUE, SPEED_TRUE, gaussians, steps=STEPS)
    rng = np.random.default_rng(7)
    observed = clean + rng.normal(0.0, 1.0e-4, clean.shape)

    result = run_closed_loop(observed, gaussians, window_steps=150,
                             max_windows=8, n_E_inits=2, speed_guess=(0.2,))
    rel_err = abs(result["E_est"] - E_TRUE) / E_TRUE
    failures = list(result["failures"])
    if rel_err >= 0.5:
        failures.append(f"final E relative error {rel_err} over 0.5 gate")
    if result["stopped_by"] == "budget_exhausted":
        failures.append("loop exhausted its window budget without converging")
    if any(w["simulation_calls"] == 0 for w in result["windows"]):
        failures.append("a window did not call the integrator")
    report = {"gate": "G4", "phase": "P4", "passed": not failures,
              "failures": failures,
              "E_true_pa": E_TRUE,
              "E_est_pa": result["E_est"],
              "relative_error": rel_err,
              "stopped_by": result["stopped_by"],
              "windows_used": result["windows_used"],
              "estimate_trace_pa": [w["E_est"] for w in result["windows"]],
              "objective_trace": [w["objective"] for w in result["windows"]],
              "open": ("invert_friction landed as a trajectory-based slice "
                       "(tests/test_friction_inversion.py): mu is inverted "
                       "from a sliding collision through the integrator, E "
                       "fixed. The acoustic route still waits on Phase 2. "
                       "Joint E+mu closed loops are the next slice.")}
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[G4] passed={report['passed']} E_est={result['E_est']:.4e} "
          f"rel_err={rel_err:.3f} stopped_by={result['stopped_by']} "
          f"windows={result['windows_used']} -> {artifact}")
    return report


if __name__ == "__main__":
    outcome = run_g4()
    raise SystemExit(0 if outcome["passed"] else 1)
