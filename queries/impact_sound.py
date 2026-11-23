"""Acoustic forward model: impact simulation -> sound waveform.

Glossary of record: tests/test_acoustic_inversion.py. ``simulate_impact_sound``
runs the real dynamics (Coulomb term included), captures the velocity jump
at contact, and synthesizes the modal sound — the same object the G2 gate
validates, packaged as a forward model for acoustic inversion.
"""
from __future__ import annotations

import numpy as np

from queries.modal_query import local_modes, synthesize_impact

DT = 1.0e-4


def simulate_impact_sound(E, speed, mu, gaussians, *, slide=0.0, dt=DT,
                          sr=44100, duration_s=1.0, steps=None,
                          approach_distance=None):
    """Simulate a sliding impact and return (waveform, meta).

    Approach along x at ``speed`` each, tangential slide along y at
    ``slide`` each (opposite signs). The integrator runs at ``dt``;
    the velocity jump across the first contact is the excitation.
    """
    from occupancy_gaussian_framework import (State, contact_pairs,
                                              langevin_step, penalty_force)
    gaussians = [type(g)(pos=g.pos.copy(), scale=g.scale.copy(), mass=g.mass,
                         tag=g.tag, E=float(E), nu=g.nu, rho=g.rho)
                 for g in gaussians]
    n = len(gaussians)
    masses = np.array([g.mass for g in gaussians])
    state = State(pos=np.stack([g.pos.copy() for g in gaussians]),
                  vel=np.zeros((n, 3)))
    state.vel[:, 0] = [speed if g.pos[0] < 0 else -speed for g in gaussians]
    state.vel[:, 1] = [slide if g.pos[0] < 0 else -slide for g in gaussians]
    pre_vel = None
    post_vel = None
    calls = 0
    steps = steps or int(round(0.3 / dt))
    # Recorded protocol: the sound starts at the impact. When the caller
    # names the initial surface gap, the approach time is prepended as
    # silence -- the impact TIMING is observable, which breaks the
    # (mu, speed) valley documented in the acoustic-inversion glossary.
    delay_s = 0.0
    if approach_distance is not None and speed > 0:
        delay_s = float(approach_distance) / (2.0 * speed)
    for _ in range(steps):
        for i in range(n):
            gaussians[i].pos = state.pos[i].copy()
        pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        force = np.zeros((n, 3))
        for pair in pairs:
            fi, fj = penalty_force(pair, gaussians, mu=mu)
            force[pair.i] += fi
            force[pair.j] += fj
        if pairs and pre_vel is None:
            pre_vel = state.vel.copy()
        if pre_vel is not None and post_vel is None and not pairs:
            post_vel = state.vel.copy()
            break
        state = langevin_step(state, force, dt, masses, T=0.0, gamma=0.0)
        calls += 1
    if pre_vel is None:
        pre_vel = state.vel.copy()
    if post_vel is None:
        post_vel = state.vel.copy()

    freqs, shapes, negative = local_modes(gaussians, list(range(n)),
                                          n_modes=6)
    dv = (post_vel - pre_vel).reshape(-1)
    if delay_s > 0:
        n_delay = int(round(delay_s * sr))
        n_total = int(duration_s * sr)
        base, bmeta = synthesize_impact(freqs, shapes, masses, dv,
                                        duration_s=max(duration_s - delay_s,
                                                         0.05), sr=sr)
        padded = np.zeros(n_total, dtype=np.float32)
        n_take = min(len(base), n_total - n_delay)
        if n_take > 0:
            padded[n_delay:n_delay + n_take] = base[:n_take]
        signal, meta = padded, bmeta
        meta["impact_delay_s"] = delay_s
    else:
        signal, meta = synthesize_impact(freqs, shapes, masses, dv,
                                         duration_s=duration_s, sr=sr)
        meta["impact_delay_s"] = 0.0
    meta["integrator_calls"] = calls
    meta["negative_clamped_count"] = negative
    return signal, meta
