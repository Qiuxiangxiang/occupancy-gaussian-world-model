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
- [ ] Rigid bodies and hinges (embedded support and mode identity are M2, not M1)
- [ ] Inverting Young's modulus from the trajectory

## M2 — embedded supports (done)

Decision, because the repo did not define it: rooted set is frozen at compile from kind `embedded`, `embedded_indices` / `embedded_objects`, and an optional ground plane. Energy is `1/2 k ||M (x-x_rest)||^2` on translational DOFs. See `PLAN.md`.

- [x] `compiler/embedded_support.py` and compile-time wiring
- [x] Active at step 0 with zero force at the rest pose, and still active with an empty contact set (`tests/test_embedded_support.py`)
- [x] `compiler/mode_tracking.py`: permutation, sign, near-degenerate Procrustes (`tests/test_mode_tracking.py`)
- [x] Embedded coast versus M1 surface `run_contact` (`compiler/m2_study.py`, `artifacts/m2_report.json`, `tests/test_m2_embedded.py`)
- [x] Smooth only for the continuous coast (max step relative change `1.06e-4` over `0.2 s`). A `0.8 m` jump relative change is `0.883`
- [x] Surface contact_on relative change `0.158`; contact_off relative change `2.67e-4` (quiet, same fact as M1)

## M3 — topology axes (not started)

Depends on M1's pair log. Identity tracking from M2 does not by itself define participation.

- [ ] Write the axis glossary in tests before adding physics: active-set bit, normal complementarity, material incompressibility, persistent embed, tangential stick/slip
- [ ] Choose units, thresholds, and hysteresis, or record them as still open
- [ ] Replay tests for the chosen schema
- [ ] Do not add a friction model beyond `penalty_force`. M1 conservation uses `mu=0`
- [ ] Modal participation is still open. Tracking exists; excitation weights do not

Underdetermined until that glossary exists. M1's predicate has no hysteresis. M2's support is not a contact bit.

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
- [x] `tests/test_embedded_support.py`
- [x] `tests/test_mode_tracking.py`
- [x] `tests/test_m2_embedded.py`

Commands that match this status:

```bash
python -c "import tests.test_harmonic as a, tests.test_contact as b, tests.test_modal as c, tests.test_langevin as d, tests.test_conservation as e; a.test_harmonic(); b.test_contact(); c.test_modal_sanity(); d.test_equipartition(); e.test_conservation()"
python -c "import tests.test_scene_compiler as t; t.main()"
python -c "import tests.test_incremental_contact as t; t.main()"
python -m compiler.harness
python -c "import tests.test_embedded_support as t; t.main()"
python -c "import tests.test_mode_tracking as t; t.main()"
python -m compiler.m2_study
```
