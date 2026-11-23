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

## M3 — topology axes (done)

Logged switches only. The penalty force is still the instantaneous 1-sigma test. Hysteresis is on the symbolic label. ``nu`` is not a switch. See `PLAN.md` for the bands.

- [x] `compiler/topology.py` normal point / area / self-penetration
- [x] Tangential sticking / sliding only when `mu > 0`; `mu = 0` is `free`
- [x] Modal participation from tracked columns, score >= 0.20 and frequency >= 1 Hz
- [x] Embedded support announced once, labeled `embedded_quadratic`, not incompressibility
- [x] Replay, hysteresis, and one fixture per axis (`tests/test_topology.py`)
- [x] `compiler/m3_study.py` and `artifacts/m3_report.json`
- [x] Observer reads `IncrementalContactGraph.active_contacts` and does not rebuild

Not done, and not claimed: complementarity, volumetric incompressibility, a second friction law.

## M4 — 1D modulus inversion (done Monday 2026-11-23)

Depends on the M1 forward contact step.

- [x] Drive `invert_young_modulus` with `langevin_step`, not a quadratic stand-in: `inference/inverse_params.py simulate_bounce` is the M1 path (contact_pairs + penalty_force + BAOAB second half-kick, T=0, gamma=0, mu=0); a test counts integrator calls
- [x] Positive or log parameterization, fixed noise seed, declared nuisances: log E clipped to bounds; observation noise at seed 7/11/3 per test; the initial approach speed is the declared nuisance, estimated jointly with E
- [x] Several initializations and an error against the simulated modulus: 6 starts (3 E x 2 speed) compared by objective; noisy recovery lands at 9.71e4 Pa vs the simulated 1e5 Pa (2.9% error), speed 0.200 vs 0.200
- [x] Keep numpy/scipy only

`inference/inverse_params.py` implements the 1D modulus slice; `invert_friction` remains a Phase 4 stub.

## M5 — telemetry (done Monday 2026-11-23, on the local RTX 4090 host)

Blocked no more: the user's RTX 4090 workstation (ea-4090-ts, reachable over Tailscale) runs the raster half. Environment: torch 2.4.1+cu124 + gsplat 1.5.3+pt24cu124 in /var/tmp/m5-telemetry (home filesystem /data is full; / has space).

- [x] Same resolution, gaussian count, warmup, repeats, explicit memory definition: 800x600, one exported 100k-gaussian PLY, LOD levels by stride decimation (only count varies), warmup 20, repeats 100, seed 0; GPU memory = torch.cuda.max_memory_allocated around the render loop
- [x] 3DGS side measured with gsplat on the 4090: 100k -> 0.482 ms median / 38.6 MB VRAM; 2.5k -> 0.236 ms / 10.2 MB (artifacts/m5_report.json); frame cost is fixed-overhead dominated below ~10k gaussians
- [x] Physics side measured on this machine: assemble_K_M is O(N^2) wall time (256 gaussians = 5.5 s), integrator step ~1e-5 s; memory via tracemalloc peak (ru_maxrss high-water overstates per-scene growth and was rejected)
- [ ] Occupancy-versus-3DGS frame comparison stays open: this repository still has no occupancy splat rasterizer, and M1 still forbids a per-frame 3DGS conversion layer

## M5 — telemetry (blocked, superseded above)

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
- [x] `tests/test_topology.py`

Commands that match this status:

```bash
python -c "import tests.test_harmonic as a, tests.test_contact as b, tests.test_modal as c, tests.test_langevin as d, tests.test_conservation as e; a.test_harmonic(); b.test_contact(); c.test_modal_sanity(); d.test_equipartition(); e.test_conservation()"
python -c "import tests.test_scene_compiler as t; t.main()"
python -c "import tests.test_incremental_contact as t; t.main()"
python -m compiler.harness
python -c "import tests.test_embedded_support as t; t.main()"
python -c "import tests.test_mode_tracking as t; t.main()"
python -m compiler.m2_study
python -c "import tests.test_topology as t; t.main()"
python -m compiler.m3_study
```

## Phase 5 — adaptive LOD (first slice done Monday 2026-11-23)

- [x] LOD pyramid by moment-preserving merges: mass/first/second moment exact, diagonal projection with PSD guard, off-diagonal residual reported per merge (`compiler/lod.py`)
- [x] Deterministic merge tree: KD-style pairing inside 1-sigma connectivity components; merges never cross gaps; disconnected elements persist; pyramid terminates instead of looping when nothing can merge
- [x] Budget policy with hysteresis as a pure function (`choose_level`); glossary of record in `tests/test_lod.py`
- [x] G5 gate (`compiler/p5_harness.py`): merge invariants over a connected grid pyramid (mass exact, scales finite) + policy replay against the real M5 4090 latency ladder
- [x] Recorded blocker: dynamics-level LOD needs P1-2 rigid reduction (no attractive bond in the integrator; compound bodies disintegrate)
- [ ] Render-side decimation smoke against the 4090 (G5c), manual when the render path is next used
- [ ] Dynamics-level LOD equivalence gates — blocked on P1-2
