"""M4: 1D Young's modulus inversion through the real forward bounce.

The forward model is the M1 contact path: `contact_pairs` + `penalty_force`
+ `langevin_step` (T=0, gamma=0). No quadratic stand-in, no curve fit that
never calls the integrator.

Recorded decisions:
- Parameterization: log E, so positivity holds by construction; clipped to
  `E_bounds` every evaluation.
- Nuisance parameter: the initial approach speed is declared a nuisance and
  estimated jointly with E (theta = (log E, speed)).
- Noise: observations are a simulated trajectory plus additive Gaussian noise
  at a fixed seed; the estimator itself is deterministic given the data.
- Initializations: several (E_init x speed_init) starts, best final objective
  wins; every start reports its own optimum.
- Report: error against the E that was actually simulated, per-start results,
  number of integrator calls.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from occupancy_gaussian_framework import (
    State,
    contact_pairs,
    langevin_step,
    penalty_force,
)

SIM_DT = 1.0e-4


def simulate_bounce(E, speed, gaussians, *, steps, dt=SIM_DT, masses=None,
                    mu=0.0):
    """Forward path: two-body approach, contact, rebound.

    Mirrors the M1 conservation settings (T=0, gamma=0, threshold 0)
    and the BAOAB second half-kick via force_fn; ``mu`` passes through to
    the Coulomb term. Returns positions of both bodies at every step,
    shape (steps, 2, 3).
    """
    pair = [type(g)(pos=g.pos.copy(), scale=g.scale.copy(), mass=g.mass,
                    tag=g.tag, E=float(E), nu=g.nu, rho=g.rho) for g in gaussians]
    if masses is None:
        masses = np.array([g.mass for g in pair], dtype=np.float64)
    state = State(pos=np.stack([g.pos for g in pair]),
                  vel=np.array([[speed, 0.0, 0.0], [-speed, 0.0, 0.0]]))
    traj = np.zeros((steps, 2, 3), dtype=np.float64)
    for step in range(steps):
        def force_fn(pos, vel=state.vel.copy()):
            for i in range(2):
                pair[i].pos = np.array(pos[i], dtype=np.float64, copy=True)
            force = np.zeros((2, 3))
            for p in contact_pairs(pair, velocities=vel, threshold=0.0):
                fi, fj = penalty_force(p, pair, mu=mu)
                force[p.i] += fi
                force[p.j] += fj
            return force

        force = force_fn(state.pos)
        state = langevin_step(state, force, dt, masses, T=0.0, gamma=0.0,
                              force_fn=force_fn)
        traj[step] = state.pos
    return traj


def invert_young_modulus(observed_trajectory, gaussians, E_bounds=(1e4, 1e9),
                         *, steps=None, speed_guess=(0.05, 0.2),
                         noise_seed=None, n_E_inits=3):
    """Estimate E (and the nuisance initial speed) from a bounce trajectory.

    observed_trajectory: (steps, 2, 3) positions, e.g. from simulate_bounce
    plus noise. gaussians: the two-body scene used for the forward model.
    Returns a report dict with the estimate, per-start optima, and the error
    against the simulated modulus when the caller supplies it via the
    returned `E_used_for_observation` field of the generating run.
    """
    observed = np.asarray(observed_trajectory, dtype=np.float64)
    if steps is None:
        steps = observed.shape[0]
    masses = np.array([g.mass for g in gaussians], dtype=np.float64)
    log_bounds = (float(np.log(E_bounds[0])), float(np.log(E_bounds[1])))
    calls = {"simulations": 0}

    def forward(theta):
        log_e, speed = float(theta[0]), float(theta[1])
        log_e = min(max(log_e, log_bounds[0]), log_bounds[1])
        speed = max(speed, 1e-6)
        calls["simulations"] += 1
        return simulate_bounce(np.exp(log_e), speed, gaussians, steps=steps,
                               masses=masses)

    def objective(theta):
        traj = forward(theta)
        diff = traj - observed
        return float(np.mean(diff ** 2))

    e_inits = np.linspace(log_bounds[0] + 0.2 * (log_bounds[1] - log_bounds[0]),
                          log_bounds[1] - 0.2 * (log_bounds[1] - log_bounds[0]),
                          int(n_E_inits))
    starts = [(log_e, float(v0)) for log_e in e_inits for v0 in speed_guess]
    results = []
    for start in starts:
        opt = minimize(objective, np.array(start), method="Nelder-Mead",
                       options={"maxiter": 40, "xatol": 1e-2, "fatol": 1e-16})
        log_e = min(max(float(opt.x[0]), log_bounds[0]), log_bounds[1])
        results.append({
            "E_init": float(np.exp(start[0])),
            "speed_init": float(start[1]),
            "E_est": float(np.exp(log_e)),
            "speed_est": float(opt.x[1]),
            "objective": float(opt.fun),
            # False means the iteration budget was exhausted, not that the
            # start failed; starts are compared by objective, not this flag.
            "converged": bool(opt.success),
        })
    best = min(results, key=lambda r: r["objective"])
    return {
        "E_est": best["E_est"],
        "speed_est": best["speed_est"],
        "objective": best["objective"],
        "E_bounds": [float(E_bounds[0]), float(E_bounds[1])],
        "nuisance": "initial approach speed, estimated jointly with E",
        "noise_seed": noise_seed,
        "simulation_calls": calls["simulations"],
        "parameterization": "log E clipped to bounds; speed floored at 1e-6",
        "starts": results,
    }


def invert_friction(observed_trajectory, gaussians, mu_bounds=(0.0, 1.0),
                    *, steps=None, speed_guess=(0.05, 0.2), E_fixed=1.0e5,
                    n_mu_inits=3, maxiter=40):
    """Estimate mu (and the nuisance initial speed) from a sliding
    collision trajectory, through the real forward model.

    Re-scoped from the Phase 4 stub (see tests/test_friction_inversion.py
    glossary): spectrum matching waits on Phase 2 acoustics; this slice
    inverts from trajectory positions like M4 inverts E. E is fixed at
    ``E_fixed`` for this slice; joint E+mu identifiability is not claimed.
    """
    observed = np.asarray(observed_trajectory, dtype=np.float64)
    if steps is None:
        steps = observed.shape[0]
    masses = np.array([g.mass for g in gaussians], dtype=np.float64)
    mu_lo, mu_hi = float(mu_bounds[0]), float(mu_bounds[1])
    calls = {"simulations": 0}

    def forward(theta):
        mu, speed = float(theta[0]), float(theta[1])
        mu = min(max(mu, mu_lo + 1e-4), mu_hi - 1e-4)
        speed = max(speed, 1e-6)
        calls["simulations"] += 1
        return simulate_bounce(E_fixed, speed, gaussians, steps=steps,
                               masses=masses, mu=mu)

    def objective(theta):
        traj = forward(theta)
        diff = traj - observed
        return float(np.mean(diff ** 2))

    mu_inits = np.linspace(mu_lo + 0.1 * (mu_hi - mu_lo),
                           mu_hi - 0.1 * (mu_hi - mu_lo), int(n_mu_inits))
    starts = [(float(m), float(v0)) for m in mu_inits for v0 in speed_guess]
    results = []
    for start in starts:
        opt = minimize(objective, np.array(start), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3,
                                "fatol": 1e-16})
        mu = min(max(float(opt.x[0]), mu_lo), mu_hi)
        results.append({
            "mu_init": float(start[0]), "speed_init": float(start[1]),
            "mu_est": float(mu), "speed_est": float(opt.x[1]),
            "objective": float(opt.fun),
            "converged": bool(opt.success),
        })
    best = min(results, key=lambda r: r["objective"])
    return {
        "mu_est": best["mu_est"],
        "speed_est": best["speed_est"],
        "objective": best["objective"],
        "mu_bounds": [mu_lo, mu_hi],
        "E_fixed_pa": E_fixed,
        "nuisance": "initial approach speed, estimated jointly; E fixed",
        "re_scope": ("spectrum matching deferred to Phase 2 acoustics; "
                     "trajectory-based inversion through the integrator"),
        "simulation_calls": calls["simulations"],
        "parameterization": "mu clipped to bounds (interior epsilon 1e-4)",
        "starts": results,
    }


def invert_joint(observed_trajectory, gaussians, E_bounds=(1e4, 1e9),
                 mu_bounds=(0.0, 1.0), *, steps=None,
                 speed_guess=(0.05, 0.2), n_inits=2, maxiter=60):
    """Joint (log E, mu, speed) inversion through the real forward model.

    Glossary of record: tests/test_joint_inversion.py. One objective sees
    both signatures (bounce timing and tangential decay). Identifiability
    is assessed, not assumed: the report carries per-start optima spread
    as the conditioning evidence.
    """
    observed = np.asarray(observed_trajectory, dtype=np.float64)
    if steps is None:
        steps = observed.shape[0]
    masses = np.array([g.mass for g in gaussians], dtype=np.float64)
    log_bounds = (float(np.log(E_bounds[0])), float(np.log(E_bounds[1])))
    mu_lo, mu_hi = float(mu_bounds[0]), float(mu_bounds[1])
    calls = {"simulations": 0}

    def forward(theta):
        log_e, mu, speed = float(theta[0]), float(theta[1]), float(theta[2])
        log_e = min(max(log_e, log_bounds[0]), log_bounds[1])
        mu = min(max(mu, mu_lo + 1e-4), mu_hi - 1e-4)
        speed = max(speed, 1e-6)
        calls["simulations"] += 1
        return simulate_bounce(np.exp(log_e), speed, gaussians, steps=steps,
                               masses=masses, mu=mu)

    def objective(theta):
        traj = forward(theta)
        diff = traj - observed
        return float(np.mean(diff ** 2))

    e_inits = np.linspace(log_bounds[0] + 0.3 * (log_bounds[1] - log_bounds[0]),
                          log_bounds[1] - 0.3 * (log_bounds[1] - log_bounds[0]),
                          int(n_inits))
    mu_inits = np.linspace(mu_lo + 0.2 * (mu_hi - mu_lo),
                           mu_hi - 0.2 * (mu_hi - mu_lo), int(n_inits))
    starts = [(float(le), float(m), float(v0))
              for le in e_inits for m in mu_inits for v0 in speed_guess]
    results = []
    for start in starts:
        opt = minimize(objective, np.array(start), method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-3,
                                "fatol": 1e-16})
        log_e = min(max(float(opt.x[0]), log_bounds[0]), log_bounds[1])
        mu = min(max(float(opt.x[1]), mu_lo), mu_hi)
        results.append({
            "E_init": float(np.exp(start[0])), "mu_init": float(start[1]),
            "speed_init": float(start[2]),
            "E_est": float(np.exp(log_e)), "mu_est": float(mu),
            "speed_est": float(opt.x[2]),
            "objective": float(opt.fun), "converged": bool(opt.success),
        })
    best = min(results, key=lambda r: r["objective"])
    good = [r for r in results
            if r["objective"] <= best["objective"] * 1.5 + 1e-20]
    logE_vals = [float(np.log(r["E_est"])) for r in good]
    mu_vals = [r["mu_est"] for r in good]
    spread = {
        "logE_std": float(np.std(logE_vals)) if len(logE_vals) > 1 else 0.0,
        "mu_std": float(np.std(mu_vals)) if len(mu_vals) > 1 else 0.0,
        "n_good_starts": len(good),
        "n_starts": len(results),
    }
    return {
        "E_est": best["E_est"],
        "mu_est": best["mu_est"],
        "speed_est": best["speed_est"],
        "objective": best["objective"],
        "E_bounds": [float(E_bounds[0]), float(E_bounds[1])],
        "mu_bounds": [mu_lo, mu_hi],
        "parameterization": "log E + bounded mu + speed; joint objective",
        "simulation_calls": calls["simulations"],
        "optima_spread": spread,
        "starts": results,
    }
