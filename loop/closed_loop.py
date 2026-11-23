"""Phase 4 closed loop: predict-observe-update with windowed inversion.

Glossary of record: tests/test_closed_loop.py. The loop receives one noisy
trajectory (the world), sees only growing prefixes, inverts each prefix
through the real forward model (M4), and stops when the E estimate stops
moving (recorded 2% relative change), when the window budget is exhausted,
or when the budget exceeds the trajectory (further windows would repeat
the same prefix — recorded as data_exhausted, no new information).
"""
from __future__ import annotations

import numpy as np

from inference.inverse_params import invert_young_modulus


def run_closed_loop(observed_trajectory, gaussians, *, window_steps: int,
                    max_windows: int, stop_relative: float = 0.02,
                    n_E_inits: int = 2, speed_guess=(0.1, 0.3),
                    noise_seed=None, invert=None) -> dict:
    """Invert growing prefixes of ``observed_trajectory``.

    Returns the belief trajectory (per-window estimates), the stopping
    decision, and the gate fields. Never reads beyond the current prefix.
    ``invert`` is the inversion entry point: callable (prefix, gaussians)
    -> report with E_est/speed_est/objective/simulation_calls; extra keys
    (e.g. mu_est from the joint estimator) are copied into the window
    records. Default: M4's invert_young_modulus.
    """
    observed = np.asarray(observed_trajectory, dtype=np.float64)
    total_steps = observed.shape[0]
    windows = []
    previous_E = None
    failures = []
    stopped_by = "budget_exhausted"
    used = 0
    for k in range(1, max_windows + 1):
        steps = min(k * window_steps, total_steps)
        if windows and steps == windows[-1]["steps"]:
            # The budget exceeds the trajectory: further windows carry no
            # new information. Stop and say so (recorded decision).
            stopped_by = "data_exhausted"
            break
        prefix = observed[:steps]
        if invert is None:
            report = invert_young_modulus(
                prefix, gaussians, n_E_inits=n_E_inits,
                speed_guess=tuple(speed_guess), noise_seed=noise_seed)
        else:
            report = invert(prefix, gaussians)
        window = {"window": k, "steps": steps,
                  "speed_est": report["speed_est"],
                  "objective": report["objective"],
                  "simulation_calls": report["simulation_calls"]}
        for extra in ("E_est", "mu_est", "optima_spread"):
            if extra in report:
                window[extra] = report[extra]
        windows.append(window)
        used = k
        # Convergence variable: E when the report carries it (default
        # entry point), otherwise mu (acoustic entry point) — recorded
        # decision: the loop tracks whatever belief the estimator emits.
        track = report.get("E_est", report.get("mu_est"))
        if track is None:
            failures.append("invert report carries neither E_est nor mu_est")
            break
        if previous_E is not None:
            change = abs(track - previous_E) / max(previous_E, 1e-30)
            if change < stop_relative:
                stopped_by = "estimate_converged"
                break
        previous_E = track
    last = windows[-1]
    estimate = last.get("E_est", last.get("mu_est"))
    if not windows:
        failures.append("no windows processed")
    return {"passed": not failures, "failures": failures,
            "E_est": estimate,
            "windows": windows,
            "windows_used": used,
            "windows_available": max_windows,
            "stopped_by": stopped_by,
            "stop_relative": stop_relative}
