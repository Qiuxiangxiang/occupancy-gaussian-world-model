"""Write the M3 example switches. The predicates live in ``compiler.topology``."""
from __future__ import annotations

import json
from pathlib import Path

from compiler.topology import example_sequences

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "m3_report.json"


def _brief(events: list[dict]) -> list[dict]:
    brief = []
    for event in events:
        brief.append({
            "seq": event["seq"],
            "time_s": event["time_s"],
            "axis": event["axis"],
            "physics": event["physics"],
            "pair_id": event.get("pair_id"),
            "before": event.get("before"),
            "after": event.get("after"),
            "volumetric_incompressibility": event.get("volumetric_incompressibility"),
        })
    return brief


def run_m3(artifact: Path = ARTIFACT) -> dict:
    sequences = example_sequences()
    report = {
        "milestone": "M3",
        "seed": None,
        "schema": "m3-constraint-switch-v1",
        "predicates": {
            "normal_enter_pen_ratio": 0.02,
            "normal_release_pen_ratio": 0.0,
            "area_enter_pen_ratio": 0.20,
            "area_release_pen_ratio": 0.12,
            "stick_enter_m_s": 1e-3,
            "stick_leave_m_s": 2e-3,
            "participation_min": 0.20,
            "participation_freq_min_hz": 1.0,
            "mu0": "free; sticking and sliding are not claimed",
            "physics": "penalty contact and the existing Coulomb term, not complementarity",
            "volumetric_incompressibility": False,
        },
        "hysteresis_chatter_events": len(sequences["hysteresis_chatter"]),
        "point_then_release": _brief(sequences["point_then_release"]),
        "area_depth": _brief(sequences["area_depth"]),
        "self_penetration": _brief(sequences["self_penetration"]),
        "mu0_free": _brief(sequences["mu0_free"]),
        "stick_slide": _brief(sequences["stick_slide"]),
        "modal_low_hz_excited": sequences["modal_low_hz_not_excited"]["excited_tracked_ids"],
        "modal_high_excited": sequences["modal_tracked_mode_excited"]["excited_tracked_ids"],
        "replay_note": "tests/test_topology.py runs each sequence twice and requires equal logs",
    }
    failures = []
    if report["hysteresis_chatter_events"] != 0:
        failures.append("chatter inside the enter band was logged")
    if [item["after"] for item in report["point_then_release"] if item["axis"] == "normal"] != ["point", "separated"]:
        failures.append(f"point sequence changed: {report['point_then_release']}")
    if [item["after"] for item in report["area_depth"] if item["axis"] == "normal"] != ["point", "area", "point"]:
        failures.append(f"area sequence changed: {report['area_depth']}")
    if [item["after"] for item in report["self_penetration"] if item["axis"] == "normal"] != ["self_penetration"]:
        failures.append("self-penetration was relabeled")
    if [item["after"] for item in report["mu0_free"] if item["axis"] == "tangential"] != ["free"]:
        failures.append("mu=0 was labeled stick or slide")
    if [item["after"] for item in report["stick_slide"] if item["axis"] == "tangential"] != ["sticking", "sliding", "sticking"]:
        failures.append(f"stick/slide sequence changed: {report['stick_slide']}")
    if report["modal_low_hz_excited"] != [] or report["modal_high_excited"] != [1]:
        failures.append("participation threshold did not match the two-mode fixture")
    report["failures"] = failures
    report["passed"] = not failures
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_m3()
    print(json.dumps({
        "passed": report["passed"],
        "failures": report["failures"],
        "point": report["point_then_release"],
        "area": report["area_depth"],
        "self": report["self_penetration"],
        "mu0": report["mu0_free"],
        "stick": report["stick_slide"],
        "modal_high": report["modal_high_excited"],
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
