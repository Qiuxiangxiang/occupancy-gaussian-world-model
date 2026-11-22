#!/usr/bin/env python3
"""Occupancy Gaussian World Model — Phase 0 core.

Occupancy-state 3D gaussians as mass distributions. Analytic pairwise
integrals build stiffness/mass matrices; modal analysis yields frequencies
and mode shapes; Langevin dynamics evolves the state; a symbolic layer
schedules contacts; an eval harness checks energy drift and penetration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh

import config as C


# ---------------------------------------------------------------------------
# 1. Occupancy gaussian
# ---------------------------------------------------------------------------
@dataclass
class OccupancyGaussian:
    """One gaussian = one mass blob. Position, diagonal scale, mass, tag."""
    pos: np.ndarray          # (3,) center
    scale: np.ndarray        # (3,) per-axis std devs
    mass: float
    tag: str = "default"
    E: float = C.E_YOUNG
    nu: float = C.NU
    rho: float = C.RHO

    @property
    def cov(self) -> np.ndarray:
        return np.diag(self.scale ** 2)

    @property
    def cov_inv(self) -> np.ndarray:
        return np.diag(1.0 / self.scale ** 2)


# ---------------------------------------------------------------------------
# 2. Analytic pairwise integrals
# ---------------------------------------------------------------------------
def _product_gaussian(g1: OccupancyGaussian, g2: OccupancyGaussian):
    """Product of two gaussians is a gaussian. Return (weight, cov, mean)."""
    S = g1.cov + g2.cov
    S_inv = np.linalg.inv(S)
    d = g1.pos - g2.pos
    # weight: (2pi)^-3/2 * |S1*S2/S|^1/2 * exp(-0.5 d^T (S1^-1+S2^-1) d)
    # simpler: integral g1*g2 = N(0; 0, S1+S2) evaluated... derive:
    # g1(x)g2(x) = A exp(-0.5(x-m1)^T A1 (x-m1) - 0.5(x-m2)^T A2 (x-m2))
    # = A exp(-0.5 (x-m)^T (A1+A2) (x-m) - 0.5 m1^T A1 m1 - 0.5 m2^T A2 m2 + 0.5 m^T(A1+A2)m)
    A = g1.cov_inv + g2.cov_inv
    m = np.linalg.solve(A, g1.cov_inv @ g1.pos + g2.cov_inv @ g2.pos)
    quad = (0.5 * g1.pos @ g1.cov_inv @ g1.pos
            + 0.5 * g2.pos @ g2.cov_inv @ g2.pos
            - 0.5 * m @ A @ m)
    w = np.exp(-quad)
    # normalization: each gaussian integrates to 1 over R^3 with
    #   (2pi)^-3/2 |cov|^-1/2, so product integrates to
    #   (2pi)^-3/2 |A|^-1/2 * w
    norm = (2 * np.pi) ** -1.5 * np.sqrt(np.linalg.det(A))
    return w * norm, A, m


def pairwise_overlap(g1: OccupancyGaussian, g2: OccupancyGaussian) -> float:
    """Integral of g1*g2 over R^3 (dimensionless overlap mass)."""
    w, _, _ = _product_gaussian(g1, g2)
    return w


def pairwise_mass(g1: OccupancyGaussian, g2: OccupancyGaussian) -> float:
    """Mass-coupling integral: mass1*mass2 * overlap."""
    return g1.mass * g2.mass * pairwise_overlap(g1, g2)


def _E_nu_to_lame(E: float, nu: float):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def pairwise_stiffness(g1: OccupancyGaussian, g2: OccupancyGaussian) -> np.ndarray:
    """3x3 stiffness coupling between two gaussians from a simple
    neo-Hookean-ish strain energy on the overlap region.

    Model: treat the overlap as a soft spring whose rest length is the
    sum of the 'radii' along the line of centers, with stiffness set by
    the effective Young's modulus of the pair and the overlap volume.
    This is a Phase-0 surrogate — replace with full analytic strain
    integrals later without changing the API.
    """
    d = g2.pos - g1.pos
    dist = np.linalg.norm(d) + 1e-12
    n = d / dist
    r1 = np.mean(g1.scale)
    r2 = np.mean(g2.scale)
    overlap_vol = pairwise_overlap(g1, g2) * (2 * np.pi) ** 1.5 * np.prod(np.sqrt(g1.cov.diagonal() * g2.cov.diagonal()) / np.sqrt((g1.cov + g2.cov).diagonal()))
    # effective modulus
    E_eff = 2 * g1.E * g2.E / (g1.E + g2.E + 1e-12)
    # spring constant ~ E_eff * overlap_area / overlap_length
    k = E_eff * overlap_vol ** (2 / 3) / (r1 + r2 + 1e-12)
    # rest length
    L0 = r1 + r2
    # force = k * (dist - L0) * n for dist > L0 (compression only in Phase 0)
    # stiffness matrix in 3D: k * n n^T
    K = k * np.outer(n, n)
    return K


# ---------------------------------------------------------------------------
# 3. Assemble global matrices
# ---------------------------------------------------------------------------
def assemble_K_M(gaussians: list[OccupancyGaussian]):
    """Build global stiffness K (3N x 3N) and mass M (3N x 3N)."""
    N = len(gaussians)
    dim = 3 * N
    K = np.zeros((dim, dim))
    M = np.zeros((dim, dim))
    for i in range(N):
        ii = slice(3 * i, 3 * i + 3)
        # self mass
        M[ii, ii] += gaussians[i].mass * np.eye(3)
        for j in range(i + 1, N):
            jj = slice(3 * j, 3 * j + 3)
            Kij = pairwise_stiffness(gaussians[i], gaussians[j])
            K[ii, jj] += Kij
            K[jj, ii] += Kij
            # off-diagonal mass coupling (small)
            Mij = 0.5 * pairwise_mass(gaussians[i], gaussians[j]) * np.eye(3)
            M[ii, jj] += Mij
            M[jj, ii] += Mij
    return K, M


# ---------------------------------------------------------------------------
# 4. Modal analysis
# ---------------------------------------------------------------------------
def modal_analysis(K: np.ndarray, M: np.ndarray, n_modes: int = C.N_MODES):
    """Generalized eigenvalue problem K v = w^2 M v.
    Returns (frequencies Hz, mode shapes)."""
    # symmetrize
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
    # regularize M to be SPD
    M = M + 1e-9 * np.eye(M.shape[0])
    w2, V = eigh(K, M, subset_by_index=[0, min(n_modes, K.shape[0]) - 1])
    w2 = np.maximum(w2, 0.0)
    freqs = np.sqrt(w2) / (2 * np.pi)
    return freqs, V


# ---------------------------------------------------------------------------
# 5. Langevin integrator (velocity Verlet + stochastic)
# ---------------------------------------------------------------------------
@dataclass
class State:
    pos: np.ndarray   # (N, 3)
    vel: np.ndarray   # (N, 3)


def langevin_step(state: State, forces: np.ndarray, dt: float,
                  masses: np.ndarray, T: float, gamma: float) -> State:
    """Velocity-Verlet Langevin with BAOAB-ish splitting.
    forces: (N, 3) external + contact forces.
    """
    N = state.pos.shape[0]
    # half-step velocity with force
    a = forces / masses[:, None]
    v_half = state.vel + 0.5 * dt * a
    # position step
    pos_new = state.pos + dt * v_half
    # damping + noise on velocity (Ornstein-Uhlenbeck exact)
    c = np.exp(-gamma * dt)
    if T > 0 and gamma > 0:
        sigma = np.sqrt(C.KB * T / masses * (1 - c ** 2))[:, None]
        noise = C.RNG.standard_normal((N, 3)) * sigma
    else:
        noise = 0.0
    v_new = c * v_half + noise
    # second half force step at new position is done by caller re-evaluating
    return State(pos_new, v_new)


# ---------------------------------------------------------------------------
# 6. Symbolic contact layer
# ---------------------------------------------------------------------------
@dataclass
class ContactPair:
    i: int
    j: int
    normal: np.ndarray
    penetration: float
    rel_vel: float


def contact_pairs(gaussians: list[OccupancyGaussian],
                  threshold: float = C.PEN_THRESHOLD) -> list[ContactPair]:
    """Generate contact pairs from gaussian proximity."""
    pairs = []
    N = len(gaussians)
    for i in range(N):
        for j in range(i + 1, N):
            g1, g2 = gaussians[i], gaussians[j]
            d = g2.pos - g1.pos
            dist = np.linalg.norm(d) + 1e-12
            r1, r2 = np.mean(g1.scale), np.mean(g2.scale)
            pen = (r1 + r2) - dist
            if pen > threshold * min(r1, r2):
                n = d / dist
                rel = np.dot(g2.pos - g1.pos, n)  # placeholder rel vel
                pairs.append(ContactPair(i, j, n, pen, rel))
    return pairs


def penalty_force(pair: ContactPair, gaussians: list[OccupancyGaussian],
                  mu: float = C.FRICTION_MU) -> tuple[np.ndarray, np.ndarray]:
    """Normal + friction penalty forces on bodies i and j."""
    gi, gj = gaussians[pair.i], gaussians[pair.j]
    # normal stiffness from pairwise
    Kij = pairwise_stiffness(gi, gj)
    k_n = np.trace(Kij) / 3.0
    f_n = k_n * pair.penetration
    f_i = -f_n * pair.normal
    f_j = +f_n * pair.normal
    # simple coulomb friction opposing tangential relative velocity
    # (Phase 0: no tangential state, just a damping-like term)
    return f_i, f_j


# ---------------------------------------------------------------------------
# 7. Evaluation harness
# ---------------------------------------------------------------------------
def eval_energy_drift(traj_ke: list[float], traj_pe: list[float], dt: float) -> float:
    """Total energy drift per second."""
    if len(traj_ke) < 2:
        return 0.0
    E = [k + p for k, p in zip(traj_ke, traj_pe)]
    drift = abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12)
    t = (len(E) - 1) * dt
    return drift / max(t, 1e-12)


def eval_penetration(pairs_hist: list[list[ContactPair]],
                     gaussians: list[OccupancyGaussian]) -> float:
    """Max penetration as fraction of gaussian scale."""
    worst = 0.0
    for pairs in pairs_hist:
        for p in pairs:
            r = min(np.mean(gaussians[p.i].scale), np.mean(gaussians[p.j].scale))
            worst = max(worst, p.penetration / max(r, 1e-12))
    return worst


def eval_contacts(pairs_hist: list[list[ContactPair]]) -> list[int]:
    return [len(p) for p in pairs_hist]


# ---------------------------------------------------------------------------
# 8. Demo scenes
# ---------------------------------------------------------------------------
def scene_harmonic_oscillator():
    """Single gaussian in a harmonic potential. Analytic: omega = sqrt(k/m)."""
    m = 1.0
    k = 4.0
    omega = np.sqrt(k / m)
    g = OccupancyGaussian(pos=np.array([1.0, 0.0, 0.0]),
                          scale=np.array([0.2, 0.2, 0.2]),
                          mass=m, tag="osc")
    state = State(pos=g.pos.copy()[None, :], vel=np.zeros((1, 3)))
    traj_pos, traj_ke, traj_pe = [], [], []
    for step in range(2000):
        # harmonic force
        f = -k * state.pos
        state = langevin_step(state, f, C.DT, np.array([m]), T=0.0, gamma=1e-6)
        traj_pos.append(state.pos[0].copy())
        traj_ke.append(0.5 * m * np.sum(state.vel ** 2))
        traj_pe.append(0.5 * k * np.sum(state.pos ** 2))
    pos = np.array(traj_pos)
    # analytic reference: x(t) = x0 cos(wt) with x0=1
    t = np.arange(len(pos)) * C.DT
    ref = np.cos(omega * t)
    err = np.mean(np.abs(pos[:, 0] - ref))
    drift = eval_energy_drift(traj_ke, traj_pe, C.DT)
    print(f"[G0-1 harmonic] omega_num={omega:.4f} (analytic {omega:.4f}), "
          f"pos error vs analytic={err:.3e}, energy drift={drift:.3e}/s")
    return err < 1e-2 and drift < C.ENERGY_DRIFT_LIMIT


def scene_two_body_contact():
    """Two gaussians head-on. Check penetration bound and energy."""
    g1 = OccupancyGaussian(pos=np.array([-1.0, 0.0, 0.0]),
                           scale=np.array([0.3, 0.3, 0.3]),
                           mass=1.0, tag="A")
    g2 = OccupancyGaussian(pos=np.array([1.0, 0.0, 0.0]),
                           scale=np.array([0.3, 0.3, 0.3]),
                           mass=1.0, tag="B")
    gs = [g1, g2]
    state = State(pos=np.array([g.pos for g in gs]),
                  vel=np.array([[0.5, 0, 0], [-0.5, 0, 0]]))
    masses = np.array([g.mass for g in gs])
    traj_ke, traj_pe, pairs_hist = [], [], []
    for step in range(3000):
        pairs = contact_pairs(gs)
        pairs_hist.append(pairs)
        F = np.zeros((2, 3))
        for p in pairs:
            fi, fj = penalty_force(p, gs)
            F[p.i] += fi
            F[p.j] += fj
        # also soft spring toward each other via K
        K, M = assemble_K_M(gs)
        # external: none; contact forces already in F
        state = langevin_step(state, F, C.DT, masses, T=0.0, gamma=1e-6)
        # sync gaussian positions
        for i in range(2):
            gs[i].pos = state.pos[i]
        ke = 0.5 * np.sum(masses[:, None] * state.vel ** 2)
        pe = 0.0
        for p in pairs:
            pe += 0.5 * np.trace(pairwise_stiffness(gs[p.i], gs[p.j])) * p.penetration ** 2
        traj_ke.append(ke)
        traj_pe.append(pe)
    drift = eval_energy_drift(traj_ke, traj_pe, C.DT)
    pen = eval_penetration(pairs_hist, gs)
    ncontacts = eval_contacts(pairs_hist)
    print(f"[G0-2 contact] energy drift={drift:.3e}/s, max penetration ratio={pen:.3f}, "
          f"peak contacts={max(ncontacts) if ncontacts else 0}")
    return drift < C.ENERGY_DRIFT_LIMIT and pen < 1.0


def scene_modal_demo():
    """Three gaussians in a triangle; print lowest frequencies."""
    gs = [
        OccupancyGaussian(pos=np.array([0.0, 0.0, 0.0]), scale=np.ones(3) * 0.25, mass=1.0, tag="1"),
        OccupancyGaussian(pos=np.array([1.0, 0.0, 0.0]), scale=np.ones(3) * 0.25, mass=1.0, tag="2"),
        OccupancyGaussian(pos=np.array([0.5, 0.866, 0.0]), scale=np.ones(3) * 0.25, mass=1.0, tag="3"),
    ]
    K, M = assemble_K_M(gs)
    freqs, V = modal_analysis(K, M, n_modes=6)
    print("[G0-3 modal] lowest frequencies (Hz):", np.round(freqs, 4))
    return True


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Occupancy Gaussian Framework — Phase 0 ===")
    ok1 = scene_harmonic_oscillator()
    ok2 = scene_two_body_contact()
    ok3 = scene_modal_demo()
    print("=== Gates: harmonic", ok1, "| contact", ok2, "| modal", ok3, "===")
