"""Cup-and-cup scene — glossary of record (written before the scene).

Recorded decisions:

- Cup definition: a ring of 8 splats (scale 0.02 m) on a circle of radius
  0.05 m in the horizontal plane. No bottom, no handle — the ring is the
  minimal cup-like object whose collision has structure (a flat disc would
  be the M1 scene again). Members overlap internally (adjacent spacing
  0.038 m < 0.04 m reach); that is fine BECAUSE the group is rigid: the
  projection absorbs internal forces, which is exactly what rigid means
  here.
- Rigidity: one explicit RigidGroup per cup (P1-2). Nothing infers the
  group from geometry.
- Collision: cups approach along x at 0.05 m/s each (closing 0.1), centers
  aligned in y and z so the rings meet splat-to-splat, dt = 1e-4 s,
  horizon 0.3 s, T = 0, gamma = 0.
- Gates: each cup's internal member separations are constant to 1e-12
  (rigid, no disintegration); a cross-cup contact occurs; total linear
  momentum is conserved to 1e-12; cross-pair penetration stays under the
  M1 limit; energy drift is reported (gated at the M1 limit, and if the
  ring geometry cannot meet it, the measured value and the decision record
  say so — no silent retuning).
"""
import numpy as np
import pytest

from scenes.cup_and_cup import (CUP_RADIUS, CUP_SCALE, N_RING, build_scene,
                                simulate)


def test_scene_shape_and_groups():
    gaussians, group_specs = build_scene()
    assert len(gaussians) == 2 * N_RING
    assert [len(spec) for spec in group_specs] == [N_RING, N_RING]
    # Ring geometry: all members of a cup share one |y| (ring plane is yz).
    left = [g.pos for g in gaussians[:N_RING]]
    radii = [float(np.linalg.norm([p[1], p[2]])) for p in left]
    assert all(r == pytest.approx(CUP_RADIUS, abs=1e-12) for r in radii)
    assert all(np.abs(p[0] - left[0][0]) < 1e-12 for p in left)


def test_collision_gate():
    result = simulate()
    assert result["passed"], result["failures"]
    assert result["contact_steps"] > 0
    assert result["momentum_drift_max"] < 1e-12
    assert result["rigidity_drift_max"] < 1e-12
    assert result["cross_penetration_ratio"] < 0.05
    # Both cups still rings afterwards: radius preserved.
    for cup in result["final_rings"]:
        for radius in cup:
            assert radius == pytest.approx(CUP_RADIUS, abs=1e-9)
