"""G1 gate: Phase 1 integration acceptance.

Scope record (PLAN.md): the original G1 was "cup + lamp hinge" (P1-1).
The hinge is deferred (no rotational DOF exists); G1 is re-scoped to the
integration acceptance of what Phase 1 actually delivered: P1-2 rigid
reduction, P1-3 non-penetration iteration, P1-4 heat accounting, on the
multi-object cup-and-cup scene, plus the friction-heat closure from the
reference scene.

G1 checks:
- cup-and-cup baseline gates (rigidity, momentum, penetration, drift);
- cup-and-cup with the non-penetration pass: rigidity/momentum intact,
  energy closure with the heat term (the recorded finding: this light
  collision stays below the projection tolerance, so heat is zero and
  closure holds trivially — reported, not hidden);
- the hard-collision reference where the projection actually engages
  (0.3 m/s head-on: max penetration x13 lower, heat accounted);
- friction-heat closure from scenes/friction_heat.py.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "g1_report.json"


def run_g1(artifact: Path = ARTIFACT) -> dict:
    from scenes.cup_and_cup import simulate as cup_simulate
    from scenes.friction_heat import collide

    failures: list[str] = []
    baseline = cup_simulate()
    if not baseline["passed"]:
        failures.extend(f"baseline: {f}" for f in baseline["failures"])
    resolved = cup_simulate(resolve=True)
    if not resolved["passed"]:
        failures.extend(f"resolved: {f}" for f in resolved["failures"])
    if resolved["projection_heat_j"] != 0.0:
        # Not a failure — but the recorded expectation for THIS light
        # collision is that the projection stays idle; a nonzero value
        # means the tolerance decision changed and the record must move.
        failures.append(
            f"resolved heat {resolved['projection_heat_j']} J is nonzero: the "
            "light collision was recorded to sit below the projection "
            "tolerance; revisit the tolerance decision record")

    friction = collide(mu=0.5)
    closure_rel = (abs(friction["closure_residual_j"])
                   / max(friction["kinetic_start_j"], 1e-30))
    if closure_rel >= 0.01:
        failures.append(f"friction heat closure {closure_rel} over 1% gate")

    # Hard-collision reference: the projection engages and helps (P1-3).
    # Numbers from the P1-3 test scenario, measured live here.
    hard = _hard_collision_reference()
    if not hard["penetration_reduced"]:
        failures.append("hard collision: projection did not reduce penetration")
    if hard["heat_j"] <= 0.0:
        failures.append("hard collision: no heat accounted")

    report = {
        "gate": "G1",
        "phase": "P1",
        "passed": not failures,
        "failures": failures,
        "scope_record": (
            "Original G1 was cup-lamp hinge (P1-1), deferred: no rotational "
            "DOF exists. G1 is re-scoped to integration acceptance of P1-2 "
            "(rigid reduction), P1-3 (non-penetration iteration), and P1-4 "
            "(heat accounting) on the cup-and-cup multi-object scene."
        ),
        "cup_baseline": {k: baseline[k] for k in (
            "passed", "contact_steps", "cross_penetration_ratio",
            "energy_drift_per_s", "momentum_drift_max",
            "rigidity_drift_max")},
        "cup_resolved": {k: resolved[k] for k in (
            "passed", "cross_penetration_ratio", "projection_heat_j",
            "closure_relative", "rigidity_drift_max",
            "momentum_drift_max")},
        "hard_collision_reference": hard,
        "friction_heat": {
            "mu": 0.5,
            "closure_relative": closure_rel,
            "heat_j": friction["friction_heat_j"],
            "tangential_rel_speed_end": friction["tangential_rel_speed_end"],
        },
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[G1] passed={report['passed']} "
          f"cup_pen={baseline['cross_penetration_ratio']:.4f} "
          f"resolved_closure={resolved['closure_relative']:.5f} "
          f"hard x{hard['reduction_factor']:.1f} -> {artifact}")
    return report


def _hard_collision_reference() -> dict:
    """0.3 m/s head-on splats with and without the projection (P1-3
    reference scenario; x13 penetration reduction measured)."""
    import numpy as np
    from compiler.nonpenetration import resolve_penetration
    from occupancy_gaussian_framework import (OccupancyGaussian, State,
                                              contact_pairs, langevin_step,
                                              penalty_force)
    dt = 1.0e-4

    def run(resolve: bool):
        gaussians = [OccupancyGaussian(pos=np.array([-0.055, 0, 0]),
                                       scale=np.ones(3) * 0.05, mass=2.36,
                                       E=1.0e5),
                     OccupancyGaussian(pos=np.array([0.055, 0, 0]),
                                       scale=np.ones(3) * 0.05, mass=2.36,
                                       E=1.0e5)]
        masses = np.array([2.36, 2.36])
        state = State(pos=np.stack([g.pos for g in gaussians]),
                      vel=np.array([[0.3, 0, 0], [-0.3, 0, 0]]))
        max_pen, heat = 0.0, 0.0
        for _ in range(3000):
            for i in range(2):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            force = np.zeros((2, 3))
            for p in pairs:
                fi, fj = penalty_force(p, gaussians, mu=0.0)
                force[p.i] += fi
                force[p.j] += fj

            def ff(pos, vel=state.vel.copy()):
                for i in range(2):
                    gaussians[i].pos = np.array(pos[i], dtype=np.float64, copy=True)
                f = np.zeros((2, 3))
                for p in contact_pairs(gaussians, velocities=vel, threshold=0.0):
                    fi, fj = penalty_force(p, gaussians, mu=0.0)
                    f[p.i] += fi
                    f[p.j] += fj
                return f

            state = langevin_step(state, force, dt, masses, T=0.0, gamma=0.0,
                                  force_fn=ff)
            for i in range(2):
                gaussians[i].pos = state.pos[i].copy()
            pairs = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
            for p in pairs:
                max_pen = max(max_pen, p.penetration)
            if resolve and pairs:
                heat += resolve_penetration(state, gaussians, pairs, masses,
                                            iterations=8, tol_m=1e-6)
        return max_pen, heat

    pen_free, _ = run(False)
    pen_fixed, heat = run(True)
    return {"without_projection_pen_m": pen_free,
            "with_projection_pen_m": pen_fixed,
            "heat_j": heat,
            "penetration_reduced": bool(pen_fixed < pen_free),
            "reduction_factor": float(pen_free / pen_fixed) if pen_fixed > 0
            else float("inf")}


if __name__ == "__main__":
    result = run_g1()
    raise SystemExit(0 if result["passed"] else 1)
