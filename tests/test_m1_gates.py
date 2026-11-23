"""M1 gates: harmonic convergence, penetration, energy drift, event log."""
from compiler.harness import run_m1


def test_m1_gates():
    report = run_m1()
    assert report["passed"], report["failures"]
    contact = report["contact"]
    assert contact["penetration_ratio"] < contact["penetration_limit"]
    assert contact["energy_drift_per_s"] < contact["energy_drift_limit_per_s"]
    assert contact["contact_steps"] > 0
    assert report["event_replay_equal"] is True
    assert report["seed"] is None


def main():
    test_m1_gates()
    print("test_m1_gates ok")


if __name__ == "__main__":
    main()
