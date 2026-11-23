# Task Checklist

`[x]` done, `[~]` partial, `[ ]` not started, `[!]` blocked.
Dates: M1 finished Sunday 2026-11-22 PST / Monday 2026-11-23 UTC. Saturday Nov 21 is not backdated.

## M1 — surface branch (done)

- [x] BAOAB second half-kick in `langevin_step` (`force_fn` for position-dependent forces)
- [x] `contact_between` shared predicate; `contact_pairs` remains the all-pairs oracle
- [x] Cross3DGS PLY read/write (`compiler/cross3dgs_ply.py`), 59-float contract cited in `PLAN.md`
- [x] Scene compiler: positions, diagonal occupancy scales, mass, Young's modulus, boundaries
- [x] Mapping table and override precedence (`compiler/materials.py`); disclaimer that color is not an identification
- [x] Incremental contact graph (`compiler/contact_graph.py`); dirty local updates; motion slack and tunnel events
- [x] `oracle_all_pairs` / `oracle_full_rebuild` for tests only (`full_rebuilds` stays 0 on the hot path)
- [x] Event log: `contact_on`, `contact_off`, `modal_spectrum` (`compiler/events.py`, `compiler/harness.py`)
- [x] G0 harness and `artifacts/m1_report.json`
- [x] Tests: `tests/test_scene_compiler.py`, `tests/test_incremental_contact.py`, `tests/test_m1_gates.py`

M1 explicitly not done, and not claimed:

- [ ] Per-frame 3DGS conversion or a splat timer
- [ ] Rigid bodies, hinges, embedded supports, mode identity
- [ ] Inverting Young's modulus from the trajectory

## M2 — embedded supports (not started)

Depends on M1's surface event log.

- [ ] Define rooted set and the support energy (missing from the repo; do not invent it silently)
- [ ] Activate that constraint at step 0 and on every later step, including steps with an empty contact set
- [ ] Track eigenmodes through permutation, sign, and near ties
- [ ] Compare the tracked embedded spectrum with the M1 surface trace
- [ ] Test that a persistent constraint stays in the active set with no collision
- [ ] Report smoothness only for states that are actually continuous. M1 already shows `contact_off` with almost no frequency change (`39.74 → 46.02 Hz` at contact on, `46.02 → 46.01 Hz` at contact off)

## M3 — three topology axes (not started)

Depends on M1's pair log. Participation also depends on M2's mode tracks.

- [ ] Write the axis glossary in tests before adding physics: active-set bit, normal complementarity, material incompressibility, persistent embed, tangential stick/slip
- [ ] Choose units, thresholds, and hysteresis, or record them as still open
- [ ] Replay tests for the chosen schema
- [ ] Do not add a friction model beyond `penalty_force`. M1 conservation uses `mu=0`

Underdetermined until that glossary exists. M1's predicate has no hysteresis.

## M4 — 1D modulus inversion (not started)

Depends on the M1 forward contact step.

- [ ] Drive `invert_young_modulus` with `langevin_step`, not a quadratic stand-in
- [ ] Positive or log parameterization, fixed noise seed, declared nuisances
- [ ] Several initializations and an error against the simulated modulus
- [ ] Keep numpy/scipy only

`inference/inverse_params.py` still raises `NotImplementedError`.

## M5 — telemetry (blocked)

- [!] Like-for-like 3DGS vs occupancy frame time and memory
- [!] Connect `ProjectTelemetry` (`android/e2e/.../ProjectTelemetry.kt` in Cross3DGS `d69f79e`) — Android battery/PSS sampler, not a renderer benchmark, and this host has no NVIDIA/Vulkan tool
- [!] Occupancy rasterizer does not exist here; do not invent timings

## Core (`occupancy_gaussian_framework.py`)

- [x] `OccupancyGaussian`
- [x] `pairwise_overlap`, `pairwise_stiffness`, `pairwise_mass`
- [x] `assemble_K_M` (coupled mass can be indefinite; M1 logs that and uses diagonal mass for those spectra)
- [x] `modal_analysis`
- [x] `langevin_step` BAOAB
- [x] `contact_between`, `contact_pairs` (all-pairs oracle)
- [x] `penalty_force` (normal plus existing Coulomb term)
- [x] `eval_energy_drift`, `eval_penetration`, `eval_contacts`
- [ ] `density_field` on the core type (compiled scenes expose `CompiledScene.density` instead)

## Config (`config.py`)

- [x] `DT`, `T`, `KB`, `DAMPING`, `RHO`, `E_YOUNG`, `NU`
- [x] `PEN_THRESHOLD`, `FRICTION_MU`, `N_MODES`
- [x] `ENERGY_DRIFT_LIMIT`, `RNG`

M1 does not change these limits. Conservation runs pass `T=0`, `gamma=0`, `mu=0` at the call, not by editing the defaults.

## Queries (`queries/modal_query.py`)

- [x] `query_frequencies` on a supplied gaussian list
- [~] `query_sound_spectrum` placeholder (equal amplitudes). Phase 2 comparison is open

## Perception (`perception/gs_to_occupancy.py`)

- [ ] gsplat reconstruction and surface resampling (Phase 3)
- [x] Import path lives in `compiler/` and is not this stub

## Inference and loop

- [ ] `inference/inverse_params.py` (M4 / Phase 4)
- [ ] `loop/closed_loop.py` (Phase 4)

## Scenes

- [x] `scenes/harmonic_oscillator.py` (demo now passes `force_fn`)
- [~] `scenes/two_body_contact.py` (runs; 0.3 s horizon still has zero contacts)
- [ ] `scenes/cup_and_lamp.py` hinge and rigid reduction

## Tests

- [x] `tests/test_harmonic.py`
- [x] `tests/test_contact.py` (legacy gate; see the zero-contact note)
- [x] `tests/test_modal.py`
- [x] `tests/test_langevin.py`
- [x] `tests/test_conservation.py`
- [x] `tests/test_scene_compiler.py`
- [x] `tests/test_incremental_contact.py`
- [x] `tests/test_m1_gates.py`

Commands that match this status:

```bash
python -c "import tests.test_harmonic as a, tests.test_contact as b, tests.test_modal as c, tests.test_langevin as d, tests.test_conservation as e; a.test_harmonic(); b.test_contact(); c.test_modal_sanity(); d.test_equipartition(); e.test_conservation()"
python -c "import tests.test_scene_compiler as t; t.main()"
python -c "import tests.test_incremental_contact as t; t.main()"
python -m compiler.harness
```
