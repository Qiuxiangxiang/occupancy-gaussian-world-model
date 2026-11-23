"""G1 gate test: Phase 1 integration acceptance runs and passes locally."""
import json

from compiler.g1_harness import run_g1


def test_g1_passes(tmp_path):
    report = run_g1(artifact=tmp_path / "g1_report.json")
    assert report["passed"], report["failures"]
    assert "scope_record" in report
    assert report["cup_baseline"]["passed"]
    assert report["cup_resolved"]["passed"]
    assert report["hard_collision_reference"]["penetration_reduced"]
    assert report["friction_heat"]["closure_relative"] < 0.01
    # The recorded finding: the light cup collision stays below the
    # projection tolerance — heat zero, closure trivial — and the gate
    # pins that record so a silent regime change fails here.
    assert report["cup_resolved"]["projection_heat_j"] == 0.0
    loaded = json.loads((tmp_path / "g1_report.json").read_text())
    assert loaded["gate"] == "G1"
