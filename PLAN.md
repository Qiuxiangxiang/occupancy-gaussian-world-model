# Implementation Plan

Six phases, 27 tasks, six validation gates.

## Phase 0 — Physics core (no vision)
- [ ] P0-1 Occupancy gaussian field: density, mass, center of mass
- [ ] P0-2 Analytic pairwise gaussian integrals (overlap, stiffness, mass)
- [ ] P0-3 Build global K and M matrices
- [ ] P0-4 Modal analysis: generalized eigenvalue solve
- [ ] P0-5 Langevin integrator (velocity Verlet + stochastic term)
- [ ] P0-6 Symbolic contact layer: pair generation, penalty force
- [ ] P0-7 Evaluation harness: energy drift, penetration, contacts
- [ ] P0-8 Demo scene: harmonic oscillator single gaussian
- [ ] P0-9 Demo scene: two-body contact
- Gate G0: three validation gates pass

## Phase 1 — Multi-object scenes
- [ ] P1-1 Cup + lamp scene with hinge constraint on lamp arm
- [ ] P1-2 Rigid body reduction: lock scales, integrate only rigid modes
- [ ] P1-3 Non-penetration solver iteration inside Langevin step
- [ ] P1-4 Energy accounting: kinetic, potential, dissipated heat
- Gate G1: cup-lamp collision conserves energy within limit

## Phase 2 — Acoustics query
- [ ] P2-1 Local submatrix extraction per object
- [ ] P2-2 Frequency + mode shape output API
- [ ] P2-3 Synthetic sound spectrum from mode superposition
- Gate G2: cup struck -> audible frequency matches FEM reference

## Phase 3 — Perception
- [ ] P3-1 3DGS reconstruction frontend (gsplat)
- [ ] P3-2 Convert appearance gaussians -> occupancy gaussians (lock positions to surface samples)
- [ ] P3-3 Thin-structure recovery (cup handle, lamp shade)
- Gate G3: reconstruction error isolated from physics error

## Phase 4 — World model (inference + prediction)
- [ ] P4-1 Forward: state read -> Langevin N steps -> symbolic contact -> render
- [ ] P4-2 Inverse: estimate Young modulus from bounce trajectory
- [ ] P4-3 Inverse: estimate friction from acoustic spectrum
- [ ] P4-4 Visual prediction: render predicted next frame
- Gate G4: parameter inversion converges from sparse noisy observations

## Phase 5 — Adaptive LOD
- [ ] P5-1 Split/merge interface with mass-momentum conservation
- [ ] P5-2 Attention-driven resolution (gaze = render priority)
- [ ] P5-3 Low-frequency physics retained in periphery
- Gate G5: no tunneling during gaze shifts; conservation error bounded

## Notes
- Phase 0-2 run on CPU; Phase 3+ want a GPU.
- Keep symbolic layer and continuous layer independently testable at every phase.
