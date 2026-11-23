"""M2: support from step 0, smooth coast, quiet surface contact_off."""
from compiler.m2_study import run_m2


def test_m2_embedded():
    report = run_m2()
    assert report["passed"], report["failures"]
    embedded = report["embedded"]
    surface = report["surface_contact"]
    assert embedded["contact_steps"] == 0
    assert embedded["support_indices"] == [0]
    assert embedded["support_evaluations"] >= embedded["steps"]
    assert report["event_log"][0]["kind"] == "embedded_support"
    assert report["event_log"][0]["since_step"] == 0
    assert surface["rel_contact_off"] < surface["rel_contact_on"]
    assert embedded["mass_basis"] == "integrator_diagonal_mass"
    assert surface["mass_basis"] == "integrator_diagonal_mass"
    assert report["seed"] is None


def main():
    test_m2_embedded()
    print("test_m2_embedded ok")


if __name__ == "__main__":
    main()
