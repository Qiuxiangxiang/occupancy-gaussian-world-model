"""P5/G5 gate: local structure. The remote render smoke (G5c) only runs
with `--remote`; here it must report as skipped, not failed."""
import json

from compiler.p5_harness import run_p5


def test_g5_local_passes_with_remote_smoke_skipped(tmp_path):
    report = run_p5(artifact=tmp_path / "p5_report.json", remote=False)
    assert report["passed"], report["failures"]
    assert report["g5a_merge_invariants"]["passed"]
    # G5b needs the M5 telemetry artifact; if present it must pass.
    g5b = report["g5b_policy_on_measured_ladder"]
    if "tight_trace" in g5b:
        assert g5b["passed"]
    assert report["g5d_dynamics_lod_equivalence"]["passed"]
    assert "note" in report["g5c_render_smoke"]
    assert (tmp_path / "p5_report.json").is_file()
    loaded = json.loads((tmp_path / "p5_report.json").read_text())
    assert loaded["gate"] == "G5"
    assert "blocked" in loaded  # the P1-2 blocker history stays recorded
