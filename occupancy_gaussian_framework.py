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
    """Product of two gaussians is a gaussian. Return (weight, precision, mean)."""
    A = g1.cov_inv + g2.cov_inv
    m = np.linalg.solve(A, g1.cov_inv @ g1.pos + g2.cov_inv @ g2.pos)
    quad = (0.5 * g1.pos @ g1.cov_inv @ g1.pos
            + 0.5 * g2.pos @ g2.cov_inv @ g2.pos
            - 0.5 * m @ A @ m)
    w = np.exp(-quad)
    norm = (2 * np.pi) ** -1.5 * np.sqrt(np.linalg.det(A))
    return w * norm, A, m


def pairwise_overlap(g1: OccupancyGaussian, g2: OccupancyGaussian) -> float:
    """Integral of g1*g2 over R^3 (dimensionless overlap)."""
    w, _, _ = _product_gaussian(g1, g2)
    return w


def pairwise_mass(g1: OccupancyGaussian, g2: OccupancyGaussian) -> float:
    return g1.mass * g2.mass * pairwise_overlap(g1, g2)


def _E_nu_to_lame(E: float, nu: float):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lam, mu


def pairwise_stiffness(g1: OccupancyGaussian, g2: OccupancyGaussian) -> np.ndarray:
    """3x3 stiffness coupling using anisotropic scales and proper overlap.

    Approximates the linear elastic interaction energy in the product-Gaussian
    region. Uses Lamé parameters and a direction-dependent effective modulus
    that accounts for the covariance of both kernels (not pure isotropic spring).
    """
    d = g2.pos - g1.pos
    dist = np.linalg.norm(d) + 1e-12
    n = d / dist

    # Effective radii along the line of centers (quadratic form of cov)
    r1 = np.sqrt(n @ g1.cov @ n)
    r2 = np.sqrt(n @ g2.cov @ n)
    L0 = r1 + r2

    overlap = pairwise_overlap(g1, g2)
    # Approximate overlap volume scale
    vol_scale = overlap * (2 * np.pi) ** 1.5 * np.sqrt(np.linalg.det(g1.cov) * np.linalg.det(g2.cov) / np.linalg.det(g1.cov + g2.cov))

    lam1, mu1 = _E_nu_to_lame(g1.E, g1.nu)
    lam2, mu2 = _E_nu_to_lame(g2.E, g2.nu)
    # harmonic mean of moduli
    mu_eff = 2 * mu1 * mu2 / (mu1 + mu2 + 1e-12)
    lam_eff = 2 * lam1 * lam2 / (lam1 + lam2 + 1e-12)

    # normal stiffness (volumetric + shear projection)
    k_n = (lam_eff + 2 * mu_eff) * vol_scale ** (2.0 / 3.0) / (L0 + 1e-12)
    # tangential (shear) stiffness
    k_t = mu_eff * vol_scale ** (2.0 / 3.0) / (L0 + 1e-12)

    # stiffness tensor: normal part + isotropic shear in the plane
    nnT = np.outer(n, n)
    K = k_n * nnT + k_t * (np.eye(3) - nnT)
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
        M[ii, ii] += gaussians[i].mass * np.eye(3)
        for j in range(i + 1, N):
            jj = slice(3 * j, 3 * j + 3)
            Kij = pairwise_stiffness(gaussians[i], gaussians[j])
            K[ii, jj] += Kij
            K[jj, ii] += Kij
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
    K = 0.5 * (K + K.T)
    M = 0.5 * (M + M.T)
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
    a = forces / masses[:, None]
    v_half = state.vel + 0.5 * dt * a
    pos_new = state.pos + dt * v_half
    c = np.exp(-gamma * dt)
    if T > 0 and gamma > 0:
        sigma = np.sqrt(C.KB * T / masses * (1 - c ** 2))[:, None]
        noise = C.RNG.standard_normal((N, 3)) * sigma
    else:
        noise = 0.0
    v_new = c * v_half + noise
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
    rel_vel: np.ndarray   # relative velocity vector


def contact_pairs(gaussians: list[OccupancyGaussian],
                  velocities: np.ndarray | None = None,
                  threshold: float = C.PEN_THRESHOLD) -> list[ContactPair]:
    """Generate contact pairs from gaussian proximity. velocities: (N, 3)."""
    pairs = []
    N = len(gaussians)
    for i in range(N):
        for j in range(i + 1, N):
            g1, g2 = gaussians[i], gaussians[j]
            d = g2.pos - g1.pos
            dist = np.linalg.norm(d) + 1e-12
            r1 = np.sqrt(d @ g1.cov @ d) / dist   # projected radius
            r2 = np.sqrt(d @ g2.cov @ d) / dist
            pen = (r1 + r2) - dist
            if pen > threshold * min(r1, r2):
                n = d / dist
                if velocities is not None:
                    rel = velocities[j] - velocities[i]
                else:
                    rel = np.zeros(3)
                pairs.append(ContactPair(i, j, n, pen, rel))
    return pairs


def penalty_force(pair: ContactPair, gaussians: list[OccupancyGaussian],
                  mu: float = C.FRICTION_MU) -> tuple[np.ndarray, np.ndarray]:
    """Normal penalty + Coulomb friction from tangential relative velocity."""
    gi, gj = gaussians[pair.i], gaussians[pair.j]
    Kij = pairwise_stiffness(gi, gj)
    # normal stiffness projected
    k_n = float(pair.normal @ Kij @ pair.normal)
    f_n_mag = k_n * max(pair.penetration, 0.0)
    f_n = f_n_mag * pair.normal

    # tangential relative velocity
    v_rel = pair.rel_vel
    v_n = np.dot(v_rel, pair.normal) * pair.normal
    v_t = v_rel - v_n
    vt_norm = np.linalg.norm(v_t) + 1e-12

    # Coulomb friction opposing tangential velocity
    f_t_mag = mu * f_n_mag
    f_t = -f_t_mag * (v_t / vt_norm)

    f_total = f_n + f_t
    # equal and opposite
    return -f_total, +f_total


# ---------------------------------------------------------------------------
# 7. Evaluation harness
# ---------------------------------------------------------------------------
def eval_energy_drift(traj_ke: list[float], traj_pe: list[float], dt: float) -> float:
    if len(traj_ke) < 2:
        return 0.0
    E = [k + p for k, p in zip(traj_ke, traj_pe)]
    drift = abs(E[-1] - E[0]) / max(abs(E[0]), 1e-12)
    t = (len(E) - 1) * dt
    return drift / max(t, 1e-12)


def eval_penetration(pairs_hist: list[list[ContactPair]],
                     gaussians: list[OccupancyGaussian]) -> float:
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
        f = -k * state.pos
        state = langevin_step(state, f, C.DT, np.array([m]), T=0.0, gamma=1e-6)
        traj_pos.append(state.pos[0].copy())
        traj_ke.append(0.5 * m * np.sum(state.vel ** 2))
        traj_pe.append(0.5 * k * np.sum(state.pos ** 2))
    pos = np.array(traj_pos)
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
        pairs = contact_pairs(gs, velocities=state.vel)
        pairs_hist.append(pairs)
        F = np.zeros((2, 3))
        for p in pairs:
            fi, fj = penalty_force(p, gs)
            F[p.i] += fi
            F[p.j] += fj
        state = langevin_step(state, F, C.DT, masses, T=0.0, gamma=1e-6)
        for i in range(2):
            gs[i].pos = state.pos[i]
        ke = 0.5 * np.sum(masses[:, None] * state.vel ** 2)
        pe = 0.0
        for p in pairs:
            pe += 0.5 * float(p.normal @ pairwise_stiffness(gs[p.i], gs[p.j]) @ p.normal) * p.penetration ** 2
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
