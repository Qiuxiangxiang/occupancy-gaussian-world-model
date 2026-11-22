# Task Checklist

Detailed tasks mapped to files. Update status as you go: [ ] todo, [~] in progress, [x] done.

## Core (`occupancy_gaussian_framework.py`)
- [ ] `OccupancyGaussian` dataclass: position, scale (diag), mass, material tag
- [ ] `density_field(x)`: sum of gaussian kernels
- [ ] `pairwise_overlap(g1, g2)`: analytic integral of product of two gaussians
- [ ] `pairwise_stiffness(g1, g2, E, nu)`: analytic strain-energy integral
- [ ] `pairwise_mass(g1, g2, rho)`: analytic mass integral
- [ ] `assemble_K_M(gaussians)`: global sparse-ish matrices
- [ ] `modal_analysis(K, M, n_modes)`: scipy eigh generalized
- [ ] `langevin_step(state, forces, dt, T, gamma)`: velocity Verlet + noise
- [ ] `contact_pairs(gaussians, threshold)`: symbolic layer pair generation
- [ ] `penalty_force(pair, penetration, mu)`: normal + friction
- [ ] `eval_energy_drift(trajectory)`: per-second drift
- [ ] `eval_penetration(trajectory)`: max depth vs scale
- [ ] `eval_contacts(trajectory)`: contact count over time

## Config (`config.py`)
- [x] DT, T, KB, DAMPING, RHO, E_YOUNG, NU
- [x] PEN_THRESHOLD, FRICTION_MU, N_MODES
- [x] ENERGY_DRIFT_LIMIT, RNG

## Queries (`queries/modal_query.py`)
- [ ] `query_frequencies(object_tag)` -> list of (freq, mode_shape)
- [ ] `query_sound_spectrum(object_tag, strike_pos, duration)` -> audio-ish array

## Perception (`perception/gs_to_occupancy.py`)
- [ ] stub: load 3DGS ply/checkpoint -> sample surface -> occupancy gaussians

## Inference (`inference/inverse_params.py`)
- [ ] stub: optimize E, nu, mu from observed trajectory / spectrum

## Loop (`loop/closed_loop.py`)
- [ ] stub: perceive -> infer -> simulate -> render cycle

## Scenes (`scenes/`)
- [ ] `harmonic_oscillator.py`: single gaussian, spring potential
- [ ] `two_body_contact.py`: two gaussians, head-on
- [ ] `cup_and_lamp.py`: rigid cup + hinged lamp arm

## Tests (`tests/`)
- [ ] `test_harmonic.py`: analytic match
- [ ] `test_contact.py`: penetration + energy
- [ ] `test_modal.py`: frequency sanity vs scale
- [ ] `test_langevin.py`: equipartition at equilibrium
- [ ] `test_conservation.py`: long-run drift bound
