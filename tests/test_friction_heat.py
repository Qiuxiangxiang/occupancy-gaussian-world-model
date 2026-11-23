"""P1-4 heat split — friction heat validation (glossary of record).

Recorded decisions:

- Heat definition: heat is accounted work removed from the mechanical
  energy by dissipative forces. Friction heat per contact step is
  ``-(f_t dot v_rel_t) * dt``, where ``f_t`` is the Coulomb force on body i
  (tangential part of the pair force) and ``v_rel_t`` is the tangential
  relative velocity. The sign convention makes heat positive: friction
  opposes relative tangential motion, so ``f_t dot v_rel_t < 0``.
- Closure gate: in a ``T=0`` collision with friction,
  ``KE_end + heat == KE_0`` up to the residual penalty potential energy and
  integrator drift. The gate tolerance is recorded (1%), not fitted.
- Momentum: every force in the scene is an internal pair force (equal and
  opposite), so total linear momentum is conserved exactly regardless of
  friction; this is asserted, not assumed.
- Damping heat (gamma > 0): still open. Langevin noise (T > 0) makes
  single-trajectory heat exact-accounting ill-defined; a fluctuation-level
  statement (average heat rate vs average KE) is the right gate there and
  is not claimed here.
"""
import numpy as np
from scenes.friction_heat import DT, collide, glossary_check


def test_glossary_scene_collides_and_heats():
    result = collide(mu=0.5)
    assert result["contact_steps"] > 0
    assert result["friction_heat_j"] > 0.0
    assert result["momentum_drift_max"] < 1e-12


def test_energy_closure_with_heat_term():
    result = collide(mu=0.5)
    closure = abs(result["kinetic_end_j"] + result["friction_heat_j"]
                  + result["potential_end_j"] - result["kinetic_start_j"])
    relative = closure / max(result["kinetic_start_j"], 1e-30)
    assert relative < 0.01, f"closure residual {relative}"


def test_friction_removes_tangential_motion():
    free = collide(mu=0.0)
    gripped = collide(mu=0.5)
    assert (abs(gripped["tangential_rel_speed_end"])
            < abs(free["tangential_rel_speed_end"]))


def test_heat_is_zero_without_friction():
    result = collide(mu=0.0)
    assert result["friction_heat_j"] == 0.0


def test_glossary_sign_convention_holds():
    """Heat is non-negative for the reference sliding collision; the
    sign convention is part of the glossary and this pins it."""
    glossary_check()
    result = collide(mu=0.5, steps=1500)
    assert result["friction_heat_j"] >= 0.0
