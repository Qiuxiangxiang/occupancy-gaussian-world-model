# Occupancy Gaussian World Model

An occupancy-state 3D Gaussian wave-packet framework for physics-based world modeling.
Represent geometry as an occupied density field (not appearance), evolve it with Langevin dynamics, schedule contacts via a symbolic layer, and query acoustics via modal analysis.

## Design

- **Occupancy Gaussians**: each Gaussian is a mass distribution, not an appearance splat. Geometry = zero level set of the density field.
- **Symbolic layer**: object identity, topology, contact pairs, material labels.
- **Continuous layer**: Langevin time integration of positions, velocities, scales.
- **Modal analysis**: local stiffness/mass matrices from analytic Gaussian integrals; eigenvalue solve gives frequencies and mode shapes (sound).
- **Closed loop**: perception -> inference -> simulation -> rendering.

## Quick start (macOS)

```bash
git clone https://github.com/Qiuxiangxiang/occupancy-gaussian-world-model.git
cd occupancy-gaussian-world-model
pip install -r requirements.txt
python occupancy_gaussian_framework.py
```

For deterministic trajectories (no thermal noise), set `T = 0.0` and `damping ~ 0` in `config.py`.

## Validation gates (Phase 0)

1. **Harmonic oscillator**: single Gaussian in a harmonic potential. Center trajectory and width pulsation must match the analytic solution; error must converge with timestep.
2. **Two-body contact**: two Gaussians generate a contact pair; check penetration depth below threshold and energy conservation (kinetic + dissipated).
3. **Baseline comparison**: same elastic-ball scenario against SPlisHSPlasH or Taichi MPM; compare stability and energy drift.

Pass criteria: energy drift < 0.1%/s, penetration < 5% of Gaussian scale, acceptable per-step cost.

## Roadmap

See `PLAN.md` for the six-phase, 27-task plan with six validation gates.
See `TASKS.md` for the detailed task checklist.
See `tests/` for the test suite.

## Modules

| Module | Role |
| --- | --- |
| `occupancy_gaussian_framework.py` | Core: field, integrals, matrices, modal analysis, Langevin, contact, eval |
| `config.py` | Parameters: temperature, damping, timestep, thresholds |
| `queries/` | Modal / acoustic query interface |
| `perception/` | 3DGS reconstruction -> occupancy Gaussians (stub) |
| `inference/` | Parameter inversion from observations (stub) |
| `loop/` | Perception-inference-simulation-rendering closed loop (stub) |
| `scenes/` | Demo scenes: cup + lamp, elastic ball |

## License

MIT
