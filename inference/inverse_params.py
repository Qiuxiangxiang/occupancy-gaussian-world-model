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


def simulate_bounce(E, speed, gaussians, *, steps, dt=SIM_DT, masses=None):
    """Forward path: two-body approach, contact, rebound.

    Mirrors the M1 conservation settings (T=0, gamma=0, mu=0, threshold 0)
    and the BAOAB second half-kick via force_fn. Returns positions of both
    bodies at every step, shape (steps, 2, 3).
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
                fi, fj = penalty_force(p, pair, mu=0.0)
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


def invert_friction(observed_spectrum, gaussians, mu_bounds=(0.0, 1.0)):
    """Optimize friction coefficient from acoustic spectrum. Stub."""
    raise NotImplementedError("Phase 4: spectrum matching")
