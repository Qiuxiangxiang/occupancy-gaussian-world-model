# Implementation Plan

Weekly milestones below are the active order. The six-phase list is the original map.
A target date is not a promise. Status is what is in this branch, not what the calendar hoped.

Clock for this status: M1 landed Sunday 2026-11-22 PST (Monday 2026-11-23 UTC).
Saturday 2026-11-21 was already past, so M1 is not marked as done on Saturday.

## M1 — surface import, G0 metrics, event log (done Sunday)

Dependency order, all done:

1. BAOAB second half-kick so a `T=0`, `gamma=0` step is velocity Verlet. The previous step applied only the first half-kick, so harmonic error did not converge and the legacy G0-1 return was false (MAE about `1.3e-2`, drift about `0.195/s`).
2. Cross3DGS PLY import and explicit export. Compiler output is an occupancy-gaussian initial condition. The file-domain payload is kept and written back; it is not a per-frame renderer.
3. Heuristic material table plus per-object and per-gaussian overrides. Appearance does not identify density or Young's modulus.
4. Incremental contact graph: dirty add/remove/move/scale updates incident edges only. All-pairs and a from-scratch rebuild are oracles for tests.
5. Harness on compiler-built scenes: harmonic convergence, penetration, energy drift, contact/frequency log.

Measured by `python -m compiler.harness` into `artifacts/m1_report.json`.
Seed is null. The run draws no RNG. Settings: `T=0 K`, `gamma=0 /s`, friction `mu=0`.

Harmonic, spring `k=4 N/m`, compiled mass about `2.362 kg`, `x0=1 m`, samples at `t=(i+1)*dt`, drift denominator `(n-1)*dt`:

| dt (s) | horizon used (s) | MAE (m) | drift (1/s) |
| --- | --- | --- | --- |
| 4e-4 | 0.1996 | 2.539e-10 | 2.247e-8 |
| 2e-4 | 0.1998 | 6.337e-11 | 5.611e-9 |
| 1e-4 | 0.1999 | 1.583e-11 | 1.402e-9 |

MAE ratios when the step is halved: about 4.006 and 4.003.

Contact, two surface splats, `dt=1e-4 s`, 4000 steps, drift horizon `0.3999 s`, `E=1e5 Pa` override (not a color guess), `rho=1200 kg/m^3`, amplitude override `1` while visual alpha is `0.5`:

- penetration ratio `0.005223` (dimensionless, `eval_penetration`), limit `PEN_THRESHOLD=0.05`
- energy drift `1.627e-6 /s`, limit `1e-3 /s`
- contact on for 41 steps, then off (not a zero-contact run)
- incremental narrowphase checks `42`, full rebuilds `0`, oracle mismatches `0`
- one-body probe: 8 bodies, 28 possible pairs, 1 narrowphase check, untouched pairs not rechecked

The penalty is armed at overlap (`threshold=0`). The legacy detector waits until penetration already exceeds 5% of scale, so it cannot be the activation used by a strict `< 0.05` ratio test. The limit itself is unchanged.

Event log (5 records, replay-equal): `modal_spectrum` at `t=0`, `contact_on` at `t=0.1001 s`, spectrum, `contact_off` at `t=0.1042 s`, spectrum. Mode identity is `not_tracked`.

Condition, not a hidden fallback: Phase 0 `pairwise_mass` made the coupled mass indefinite (minimum eigenvalue about `-948` at the start). Logged frequencies use the integrator's diagonal masses. Nonzero frequencies moved from about `39.74/74.34 Hz` at contact-on time's previous sample to `46.02/86.10 Hz` when the contact turned on, and only to `46.01/86.07 Hz` when it turned off. The off event is not a spectral jump: stiffness follows overlap, while the symbolic bit is a threshold. M2 must not assume every surface event jumps, and must not assume an embedded constraint is smooth, until the state under test is actually continuous.

Known M1 limits:

- No object ids in a Cross3DGS PLY, so the gate scene is unsegmented independent particles. Segmentation in a sidecar labels and assigns materials. It does not lock a rigid shape.
- Diagonal world-axis scales are what Phase 0 integrates. Oriented ellipsoids stay in the exchange quaternion.
- Legacy `scene_two_body_contact` still simulates `0.3 s` from a `2 m` gap and records zero contacts. It is not the M1 gate.
- Mesh import is not implemented. The perception stub still does not reconstruct from images.

## M2 — embedded supports (done in this branch)

The repo never named a rooted set or a support energy. The rule that is implemented, and frozen at compile time, is the union of:

- boundary kind `embedded`
- `embedded_indices` / `embedded_objects`
- centers on the support side of an optional `ground_plane` (`normal · x <= offset`)

Later motion does not add or remove members. A surface contact does not create a support.

Energy, for each rooted gaussian, on the translational DOFs Phase 0 stores (there is no rotational DOF):

`E = 1/2 k || M (x - x_rest) ||^2` joules, `F = -k M (x - x_rest)` newtons, with `x` in meters and `k` in N/m. `M` is a 0/1 mask, default all three axes. `k` is `support_stiffness` when the sidecar sets it, otherwise `E_young * mean(scale)` (Pa·m = N/m). That product is a default, not a calibration. `x_rest` is the compiled position.

The force is added on every evaluation passed to `langevin_step`, including the second kick, from step 0. At the rest pose the force and energy are zero and the stiffness is still in a copy of `K` (`add_to_stiffness` after `assemble_K_M`; that function is not edited). `apply_boundary_conditions` does not move these bodies. A fixed boundary remains a Dirichlet reset and is not this support.

Measured by `python -m compiler.m2_study` into `artifacts/m2_report.json`. Seed is null. No RNG draw. Mass basis for both runs is `integrator_diagonal_mass` because the coupled mass is indefinite, the same condition as M1.

Embedded regime: rooted mass `1 kg`, free mass `1 kg`, `k = 5000 N/m`, `dt = 1e-4 s`, 2000 steps, horizon `0.2 s`. The free body coasts from `0.16 m` toward the root and stops near `0.14 m`, outside the `0.10 m` contact distance. Contact steps: `0`. Support evaluations: `4001` (the residual at step 0 plus both kicks of every step). Indices stay `[0]`. Full contact rebuilds: `0`.

Tracked nonzero frequencies (Rayleigh, identity tracked) move from `42.807 / 79.066 Hz` to `51.977 / 96.409 Hz`. Total relative change about `0.219`. Largest single-step relative change about `1.06e-4`. A teleport of the free body to `0.8 m` changes the tracked frequencies by about `0.883` in one step and drops the minimum overlap to about `0.71`. Smoothness is this continuous coast, not a property of every pose change.

Surface baseline is a fresh `run_contact` (M1 path, sorted eigenvalues, identity not tracked):

- contact on at `0.1001 s`: `39.737 / 74.341 Hz` to `46.020 / 86.095 Hz`, relative change `0.158`
- contact off at `0.1042 s`: to `46.008 / 86.072 Hz`, relative change `2.67e-4`

The off event remains spectrally quiet. The on event is much larger than one embedded smooth step. Mode identity for the embedded run uses overlap in the mass inner product, Hungarian assignment, and a Procrustes rotation inside frequency clusters closer than `0.01 Hz + 0.1%`. Sign flips and subspace rotations are recorded. M1's own log is left as `not_tracked`.

## M3 — topology axes (done)

"Incompressibility constraint switching" here means an active-set change of a constraint the code already has. It is not a volumetric incompressibility switch. `nu` stays a Lamé parameter. The contact force remains the penalty in `penalty_force`, not a complementarity condition. Every `constraint_switch` sets `volumetric_incompressibility` to false and names its `physics`.

The instantaneous 1-sigma predicate still decides the force, so G0 is unchanged. The symbolic label uses a Schmitt band and is written only when that label changes. Pair id is `"{i}-{j}"` with `i < j`. Support id is `support:{index}`. `seq` is the log order and the replay key.

Normal axis, penetration ratio = penetration / min(mean scale), dimensionless:

- enter at ratio >= 0.02, release only at ratio <= 0
- `point`: one logged pair below the area band
- `area`: ratio >= 0.20, held until ratio < 0.12, or at least two logged pairs between the same object pair
- `self_penetration`: both ends share a non-null object id. Unsegmented ids do not count

Tangential axis, speed in m/s, same Coulomb term, no second friction model:

- `free` when `mu <= 0` while the normal label is on. Sticking is undefined. This is not called sliding
- `sticking` when `mu > 0` and speed <= 1e-3 m/s
- `sliding` when `mu > 0` and speed >= 2e-3 m/s
- between 1e-3 and 2e-3 the previous stick or slide label stays

Modal axis, on a switch only: `score_i = (φ_i · f)^2 / sum_j (φ_i · f)^2` using tracked columns. Excited when score >= 0.20 and frequency >= 1 Hz. A 0.2 Hz mode with all the force is not excited.

Embedded supports are announced once as `embedded_quadratic`. They are not re-logged every step.

Example, seed null, from `python -m compiler.m3_study`:

- ratios 0, 0.01, 0, 0.015, 0 produce no events
- ratio 0.05 then -0.05: normal `separated → point` at 0.01 s, tangential `separated → free`, then both return to `separated` at 0.02 s
- ratio 0.05, 0.25, 0.05: normal `point → area → point`. The middle value 0.15 would have stayed `area` (release is 0.12); 0.05 is below that
- same object id: normal `self_penetration`, physics `penalty_contact`
- `mu = 0` and tangential speed 1 m/s: tangential `free`
- `mu = 0.3`: speed 0 `sticking`, 5e-3 m/s `sliding`, 1.5e-3 m/s no event, speed 0 `sticking` again
- modal force on the 10 Hz tracked mode: excited id `[1]`; the same force on a 0.2 Hz mode: excited `[]`

## M4 — 1D Young's modulus order-of-magnitude inversion (not started)

Depends on a real forward bounce. M1's contact run is that forward path (`langevin_step`, not a stand-in quadratic). `inference/inverse_params.py` is still the Phase 4 stub.

When it starts: positive or log parameterization, deterministic noise, stated nuisance parameters, more than one initialization, and an error against the modulus that was actually simulated. Stay on numpy/scipy. Do not report a curve fit that never calls the integrator.

## M5 — telemetry (blocked)

Not started, and not measurable here.

What was checked:

- Public Cross3DGS commit `d69f79e1431d39d7ba71e22c29065dca67a4b4ce` (2026-11-18). `ProjectTelemetry.kt` is an Android E2E sampler (process PSS, native heap, thermal status, battery). `E2eTelemetry.mm` is the iOS counterpart. Neither is a 3DGS-versus-occupancy frame benchmark.
- `scripts/benchmark_android.py`, `benchmark_ios.py`, `benchmark_brush.py`, `benchmark_vksplat.py` target devices and other splat runners. This machine has no `nvidia-smi` and no `vulkaninfo`.
- This repository has no occupancy splat rasterizer. M1 forbids a per-frame 3DGS conversion layer, and none was added.
- The user's GitHub account lists this repository and not a second Cross3DGS checkout. Origin namespaces visible to this run are empty.

No latency or memory number is filled in. A later run needs the same resolution, gaussian count, warmup, repeats, and an explicit CPU-versus-GPU memory definition on hardware that can execute both renderers.

## Exchange contract (verified)

Writer: `libs/cross3dgs/src/gaussian_state.cpp` `SaveGaussianPly`.
Check: `tests/cpu/test_ply_export.cpp` (59 floats, `f_rest` channel-major).

- binary little-endian, 59 `float` properties, no normals
- `opacity` logit, `scale_*` log, `rot_0..3` = `(w, x, y, z)`
- `f_rest` index `channel * 15 + coefficient`
- `kShC0 = 0.28209479177387814`

Inria-style `nx, ny, nz` files are a different layout. Named extras can be ignored on read. The writer emits the 59-field header only. Internal `.npz` / dict input must name its conventions; it is not the exchange schema.

## Original phases

### Phase 0 — physics core

- [x] P0-1 Occupancy gaussian field (lumped mass; density of the compiled kernels lives on `CompiledScene.density`)
- [x] P0-2 Analytic pairwise integrals
- [x] P0-3 Global K and M
- [x] P0-4 Modal analysis (`eigh`), with the M1 caveat that coupled M can be indefinite
- [x] P0-5 Langevin / BAOAB integrator
- [x] P0-6 Symbolic contact predicate and penalty
- [x] P0-7 Evaluation harness (`eval_energy_drift`, `eval_penetration`, `eval_contacts`) plus the M1 compiler scenes
- [x] P0-8 Harmonic demo
- [x] P0-9 Two-body demo (legacy horizon still misses contact; M1 is the contact gate)
- [x] Gate G0 on the M1 scenes

### Phase 1 — multi-object scenes

- [x] Surface-contact initial conditions from a Cross3DGS cloud (M1)
- [x] Embedded quadratic support, frozen at compile, active with an empty contact set (M2)
- [ ] P1-1 Cup + lamp hinge
- [ ] P1-2 Rigid reduction
- [ ] P1-3 Non-penetration iteration inside the step
- [ ] P1-4 Full energy split into kinetic, potential, and heat
- [ ] Gate G1 cup-lamp

### Phase 2 — acoustics

- [x] Minimal `query_frequencies` / placeholder spectrum in `queries/modal_query.py`
- [ ] Local submatrix per object, tracked modes, audible comparison to a FEM reference
- [ ] Gate G2

### Phase 3 — perception

- [x] Import of an existing Cross3DGS cloud (this is not reconstruction)
- [ ] gsplat frontend, surface resampling, thin structures
- [ ] Gate G3

### Phase 4 — world model

- [ ] Forward closed loop and inverse E / friction (`inference/inverse_params.py`, `loop/closed_loop.py` still raise)
- [ ] M4 is the 1D modulus slice of P4-2, not the whole phase
- [ ] Gate G4

### Phase 5 — adaptive LOD

- [ ] Unchanged. Not in the weekly milestones.

## Rendering

Occupancy gaussians are the draw primitive. Do not insert a 3DGS point-cloud conversion on the frame path. `write_cross3dgs_ply` is an explicit export of the file-domain payload.
