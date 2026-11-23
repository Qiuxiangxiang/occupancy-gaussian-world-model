"""Embedded-support frequency study versus the M1 surface-contact log.

The smooth regime is a rooted gaussian and a free gaussian that coasts toward
it without entering the 1-sigma contact predicate. Support stiffness is in K
and in the force from step 0. A single position jump is reported separately
and is not counted as smooth evolution.

Surface numbers come from ``compiler.harness.run_contact``, the M1 path.
Sorted eigenvalues there are not mode-tracked. The embedded run tracks
identity. Comparison uses the nonzero frequencies in both.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

import config as C
from compiler.contact_graph import IncrementalContactGraph
from compiler.events import EventLog
from compiler.harness import run_contact
from compiler.mode_tracking import ModeTracker
from compiler.scene_compiler import compile_scene
from occupancy_gaussian_framework import (
    State,
    assemble_K_M,
    langevin_step,
    modal_analysis,
    penalty_force,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "m2_report.json"

DT = 1.0e-4
STEPS = 2000
FREE_SPEED = -0.1  # m/s, toward the rooted body along +x but starting outside contact


def _modal(gaussians, supports, locked_basis: str | None):
    stiffness, mass = assemble_K_M(gaussians)
    stiffness = supports.add_to_stiffness(stiffness)
    coupled = 0.5 * (mass + mass.T)
    minimum = float(np.linalg.eigvalsh(coupled)[0])
    spd = minimum > 0.0
    if locked_basis is None:
        locked_basis = "phase0_consistent_mass" if spd else "integrator_diagonal_mass"
    if locked_basis == "phase0_consistent_mass":
        used = mass
    else:
        used = np.zeros_like(mass)
        for index, gaussian in enumerate(gaussians):
            slot = slice(3 * index, 3 * index + 3)
            used[slot, slot] = gaussian.mass * np.eye(3)
    count = min(int(C.N_MODES), stiffness.shape[0])
    freqs, vectors = modal_analysis(stiffness, used, n_modes=max(count, 1))
    return freqs, vectors, stiffness, used, locked_basis, spd, minimum


def _rel_change(current, previous, floor_hz: float = 1.0) -> float:
    worst = 0.0
    for new, old in zip(current, previous):
        if abs(old) < floor_hz:
            continue
        worst = max(worst, abs(new - old) / max(abs(old), floor_hz))
    return float(worst)


def _compile_embedded():
    # V = (2π)^{3/2} * scale^3. rho = 1/V makes the lumped mass 1 kg at amplitude 1.
    scale = 0.05
    volume = (2.0 * np.pi) ** 1.5 * scale ** 3
    rho = 1.0 / volume
    scene = compile_scene({
        "xyz": np.array([[0.0, 0.0, 0.0], [0.16, 0.0, 0.0]]),
        "scale": np.ones((2, 3)) * scale,
        "opacity": np.array([0.5, 0.5]),
        "conventions": {"opacity": "alpha", "scale": "linear", "quaternion": "wxyz"},
        "metadata": {
            "length_unit": "m",
            "gaussian_overrides": {
                "0": {"E": 1.0e5, "rho": rho, "nu": 0.3, "occupancy_amplitude": 1.0},
                "1": {"E": 1.0e5, "rho": rho, "nu": 0.3, "occupancy_amplitude": 1.0},
            },
            "embedded_indices": [0],
            "support_stiffness": 5000.0,
            "initial_velocity": [[0.0, 0.0, 0.0], [FREE_SPEED, 0.0, 0.0]],
        },
    })
    return scene


def run_embedded() -> dict:
    scene = _compile_embedded()
    gaussians = scene.gaussians
    supports = scene.supports
    assert supports.indices() == [0]
    state = State(
        pos=np.stack([gaussian.pos.copy() for gaussian in gaussians]),
        vel=scene.initial_velocities.copy(),
    )
    masses = np.array([gaussian.mass for gaussian in gaussians])
    log = EventLog()
    graph = IncrementalContactGraph(threshold=0.0, log=log)
    for index, gaussian in enumerate(gaussians):
        graph.add(index, gaussian)
    graph.commit(gaussians, 0.0, state.vel)
    # Step 0, before any motion and with an empty contact set.
    assert graph.active_contacts() == []
    rest_energy = supports.energy(state.pos)
    rest_force = supports.force(state.pos)
    log.append(
        0.0, "embedded_support",
        indices=supports.indices(),
        stiffness_N_per_m=[item.stiffness_Npm for item in supports.supports],
        dof_mask=[item.mask.tolist() for item in supports.supports],
        rest_m=[item.rest_m.tolist() for item in supports.supports],
        reasons=[list(item.reasons) for item in supports.supports],
        energy_J=rest_energy,
        force_N=rest_force.tolist(),
        active=True,
        since_step=0,
        contact_pairs=0,
    )
    freqs, vectors, stiffness, mass, basis, initial_spd, minimum = _modal(gaussians, supports, None)
    tracker = ModeTracker()
    first = tracker.update(freqs, vectors, mass, stiffness)
    log.append(
        0.0, "modal_spectrum",
        frequencies_hz=first["tracked_frequencies_hz"],
        sorted_frequencies_hz=first["sorted_frequencies_hz"],
        ordering="tracked_rayleigh",
        mode_identity="tracked",
        mass_basis=basis,
        consistent_mass_spd=initial_spd,
        consistent_mass_min_eigenvalue=minimum,
        reason="embedded_initial",
        min_overlap=first["min_overlap"],
    )
    max_step = 0.0
    previous = first["tracked_frequencies_hz"]
    started = previous
    contact_steps = 0
    for step in range(STEPS):
        def force_fn(pos, vel=state.vel):
            for index in range(len(gaussians)):
                gaussians[index].pos = np.array(pos[index], dtype=np.float64, copy=True)
                if not np.array_equal(gaussians[index].pos, graph.bodies[index].pos):
                    graph.mark_moved(index)
            graph.commit(gaussians, (step + 1) * DT, vel)
            force = supports.force(pos)
            for pair in graph.active_contacts():
                left, right = penalty_force(pair, gaussians, mu=0.0)
                force[pair.i] += left
                force[pair.j] += right
            return force

        state = langevin_step(
            state, force_fn(state.pos), DT, masses, T=0.0, gamma=0.0, force_fn=force_fn,
        )
        if graph.active_contacts():
            contact_steps += 1
        assert supports.indices() == [0]
        freqs, vectors, stiffness, mass, _basis, _spd, _minimum = _modal(
            gaussians, supports, basis,
        )
        report = tracker.update(freqs, vectors, mass, stiffness)
        max_step = max(max_step, _rel_change(report["tracked_frequencies_hz"], previous))
        previous = report["tracked_frequencies_hz"]
    total = _rel_change(previous, started)
    log.append(
        STEPS * DT, "modal_spectrum",
        frequencies_hz=previous,
        ordering="tracked_rayleigh",
        mode_identity="tracked",
        mass_basis=basis,
        reason="embedded_smooth_end",
        max_rel_step=max_step,
        total_rel=total,
    )
    log.append(
        STEPS * DT, "embedded_support",
        indices=supports.indices(),
        active=True,
        since_step=0,
        contact_pairs=contact_steps,
        evaluations=supports.evaluations,
    )
    # Discontinuous pose, not part of the smooth statistic.
    gaussians[1].pos = np.array([0.8, 0.0, 0.0])
    graph.mark_moved(1)
    graph.commit(gaussians, STEPS * DT, state.vel)
    freqs, vectors, stiffness, mass, basis, spd, minimum = _modal(gaussians, supports, basis)
    jumped = tracker.update(freqs, vectors, mass, stiffness)
    jump = _rel_change(jumped["tracked_frequencies_hz"], previous)
    return {
        "dt_s": DT,
        "steps": STEPS,
        "horizon_s": STEPS * DT,
        "seed": None,
        "mass_kg": [float(gaussian.mass) for gaussian in gaussians],
        "mass_basis": basis,
        "consistent_mass_spd_initial": bool(initial_spd),
        "support_indices": supports.indices(),
        "support_stiffness_N_per_m": [item.stiffness_Npm for item in supports.supports],
        "support_evaluations": supports.evaluations,
        "support_active_every_sampled_step": True,
        "contact_steps": contact_steps,
        "full_rebuilds": graph.full_rebuilds,
        "tracked_hz_start": started,
        "tracked_hz_end": previous,
        "max_rel_step": max_step,
        "total_rel": total,
        "jump_rel": jump,
        "jump_min_overlap": jumped["min_overlap"],
        "rest_energy_J": rest_energy,
        "rest_force_norm_N": float(np.linalg.norm(rest_force)),
        "events": log.events,
    }


def _surface_transition(events: list[dict]) -> dict:
    spectra = [event for event in events if event["kind"] == "modal_spectrum"]
    topology = [event for event in events if event["kind"] in ("contact_on", "contact_off")]
    by_time = {event["time_s"]: event for event in spectra if event["reason"] == "contact_topology"}
    initial = next(event for event in spectra if event["reason"] == "initial")
    switched_on = next(event for event in topology if event["kind"] == "contact_on")
    switched_off = next(event for event in topology if event["kind"] == "contact_off")
    on_spec = by_time[switched_on["time_s"]]
    off_spec = by_time[switched_off["time_s"]]
    return {
        "source": "compiler.harness.run_contact",
        "mode_identity": "not_tracked",
        "mass_basis": initial["mass_basis"],
        "dt_s": 1.0e-4,
        "contact_on_time_s": switched_on["time_s"],
        "contact_off_time_s": switched_off["time_s"],
        "hz_before_contact": initial["frequencies_hz"],
        "hz_at_contact_on": on_spec["frequencies_hz"],
        "hz_at_contact_off": off_spec["frequencies_hz"],
        "rel_contact_on": _rel_change(on_spec["frequencies_hz"], initial["frequencies_hz"]),
        "rel_contact_off": _rel_change(off_spec["frequencies_hz"], on_spec["frequencies_hz"]),
    }


def run_m2(artifact: Path = ARTIFACT) -> dict:
    rng_before = C.RNG.bit_generator.state
    embedded = run_embedded()
    with tempfile.TemporaryDirectory() as tmp:
        surface_run = run_contact(Path(tmp))
    surface = _surface_transition(surface_run["events"])
    rng_after = C.RNG.bit_generator.state
    failures = []
    if embedded["support_indices"] != [0]:
        failures.append("rooted set is not the compiled index 0")
    if embedded["contact_steps"] != 0:
        failures.append("smooth embedded run produced a surface contact")
    if embedded["support_evaluations"] < embedded["steps"]:
        failures.append("support force was not evaluated across the horizon")
    if embedded["rest_force_norm_N"] > 1e-8 or abs(embedded["rest_energy_J"]) > 1e-10:
        failures.append("support at the rest pose should be active with zero force and energy")
    if embedded["max_rel_step"] >= 1e-3:
        failures.append(f"embedded per-step frequency change {embedded['max_rel_step']} is not smooth")
    if embedded["total_rel"] <= 0.05:
        failures.append(f"embedded spectrum did not evolve ({embedded['total_rel']})")
    if embedded["jump_rel"] <= 0.2 or embedded["jump_rel"] < 100.0 * embedded["max_rel_step"]:
        failures.append(
            f"discontinuous jump {embedded['jump_rel']} was not separated from the smooth step "
            f"{embedded['max_rel_step']}"
        )
    if surface["rel_contact_on"] <= embedded["max_rel_step"] * 10.0:
        failures.append("surface contact_on change is not larger than the embedded smooth step")
    if surface["rel_contact_off"] >= surface["rel_contact_on"] * 0.1:
        failures.append(
            "surface contact_off moved about as much as contact_on; "
            "M1's quiet contact_off baseline was lost"
        )
    kinds = [event["kind"] for event in embedded["events"]]
    if "contact_on" in kinds or "contact_off" in kinds:
        failures.append("embedded event log recorded a contact topology change")
    if kinds[0] != "embedded_support" or embedded["events"][0]["since_step"] != 0:
        failures.append("support event is not the first record at step 0")
    if embedded["full_rebuilds"] != 0:
        failures.append("embedded run rebuilt the contact graph")
    if rng_before != rng_after:
        failures.append("M2 study consumed the global RNG")
    report = {
        "milestone": "M2",
        "seed": None,
        "rng_consumed": rng_before != rng_after,
        "embedded": {key: value for key, value in embedded.items() if key != "events"},
        "surface_contact": surface,
        "event_log": embedded["events"],
        "failures": failures,
        "passed": not failures,
        "reading": (
            "Embedded frequencies move continuously while the free body coasts "
            "and the support set stays {0}. A position jump does not. Surface "
            "contact_on changes the sorted spectrum by much more than one embedded "
            "step; contact_off, as in M1, barely moves it because overlap is continuous."
        ),
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_m2()
    embedded = report["embedded"]
    surface = report["surface_contact"]
    print(json.dumps({
        "passed": report["passed"],
        "failures": report["failures"],
        "mass_basis": embedded["mass_basis"],
        "max_rel_step": embedded["max_rel_step"],
        "total_rel": embedded["total_rel"],
        "jump_rel": embedded["jump_rel"],
        "support_evaluations": embedded["support_evaluations"],
        "contact_steps": embedded["contact_steps"],
        "surface_rel_on": surface["rel_contact_on"],
        "surface_rel_off": surface["rel_contact_off"],
        "hz_start": embedded["tracked_hz_start"],
        "hz_end": embedded["tracked_hz_end"],
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
