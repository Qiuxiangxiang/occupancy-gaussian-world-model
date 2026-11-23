"""M1 regression: compiler scenes through the Phase 0 integrator.

Conservation uses T = 0 and gamma = 0. Langevin noise and damping are not
energy-conserving, and the contact run also sets friction to 0 because the
Coulomb term in ``penalty_force`` dissipates. The penetration limit is
``config.PEN_THRESHOLD`` (dimensionless penetration / min mean scale, the
quantity ``eval_penetration`` already returns). The legacy detector waits
until penetration already exceeds that fraction, so this run arms the penalty
at first overlap (threshold 0) and still requires the measured ratio to stay
below the configured limit.

Contact forces come from the incremental graph. ``contact_pairs`` is called
only as an oracle.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as C
from compiler.contact_graph import IncrementalContactGraph, oracle_all_pairs
from compiler.cross3dgs_ply import SH_C0, cloud_from_activated, write_cross3dgs_ply
from compiler.events import EventLog
from compiler.scene_compiler import compile_scene
from occupancy_gaussian_framework import (
    State,
    assemble_K_M,
    contact_pairs,
    eval_energy_drift,
    eval_penetration,
    langevin_step,
    modal_analysis,
    pairwise_stiffness,
    penalty_force,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts" / "m1_report.json"

CONTACT_DT = 1.0e-4
CONTACT_HORIZON = 0.4
CONTACT_SPEED = 0.1
HARMONIC_K = 4.0
HARMONIC_HORIZON = 0.2
HARMONIC_DTS = (4.0e-4, 2.0e-4, 1.0e-4)


def _rgb_dc(rgb):
    return (np.asarray(rgb, dtype=np.float64) - 0.5) / SH_C0


def _write_fixture(directory: Path, name: str, xyz, scale, alpha, velocity, overrides):
    n = len(xyz)
    f_dc = np.repeat(_rgb_dc((0.80, 0.15, 0.15))[None, :], n, axis=0)
    quat = np.zeros((n, 4), dtype=np.float64)
    quat[:, 0] = 1.0
    cloud = cloud_from_activated(xyz, scale, alpha, quat, f_dc)
    ply = directory / f"{name}.ply"
    write_cross3dgs_ply(ply, cloud)
    meta = {
        "length_unit": "m",
        "unit_scale": 1.0,
        "gaussian_overrides": {
            str(i): dict(overrides) for i in range(n)
        },
        "initial_velocity": np.asarray(velocity, dtype=np.float64).tolist(),
    }
    (directory / f"{name}.json").write_text(json.dumps(meta), encoding="utf-8")
    return ply


def _spectrum(gaussians):
    """Frequencies from Phase 0's K and, when it is SPD, its coupled M.

    ``pairwise_mass`` is not a Gram matrix of normalized kernels. For splats
    whose overlap makes the coupled mass indefinite, the log uses the diagonal
    masses the integrator actually integrates and says so. That is a recorded
    condition, not a claim that the coupled eigenproblem succeeded.
    """
    if not gaussians:
        return [], "empty", True, None
    stiffness, mass = assemble_K_M(gaussians)
    coupled = 0.5 * (mass + mass.T)
    minimum = float(np.linalg.eigvalsh(coupled)[0])
    count = min(int(C.N_MODES), int(stiffness.shape[0]))
    if minimum > 0.0:
        basis = "phase0_consistent_mass"
        used = mass
    else:
        basis = "integrator_diagonal_mass"
        used = np.zeros_like(mass)
        for index, gaussian in enumerate(gaussians):
            slot = slice(3 * index, 3 * index + 3)
            used[slot, slot] = gaussian.mass * np.eye(3)
    freqs, _ = modal_analysis(stiffness, used, n_modes=max(count, 1))
    return [float(freq) for freq in freqs], basis, minimum > 0.0, minimum


def _append_spectrum(log: EventLog, gaussians, time_s: float, reason: str) -> None:
    freqs, basis, spd, minimum = _spectrum(gaussians)
    log.append(
        time_s, "modal_spectrum",
        frequencies_hz=freqs,
        ordering="eigenvalue_ascending",
        mode_identity="not_tracked",
        mass_basis=basis,
        consistent_mass_spd=spd,
        consistent_mass_min_eigenvalue=minimum,
        reason=reason,
        note=(
            "Sorted generalized eigenvalues at this pose. Mode identity across "
            "steps is not tracked. A contact event is not required to produce "
            "a spectral jump. If the Phase 0 coupled mass is indefinite, the "
            "frequencies use the integrator's diagonal masses."
        ),
    )


def _pair_key(pair):
    return (int(pair.i), int(pair.j))


def _same_contacts(got, ref) -> bool:
    if [_pair_key(pair) for pair in got] != [_pair_key(pair) for pair in ref]:
        return False
    for left, right in zip(got, ref):
        if abs(float(left.penetration) - float(right.penetration)) > 1e-8:
            return False
        if float(np.linalg.norm(left.normal - right.normal)) > 1e-8:
            return False
    return True


def _forces(graph, gaussians):
    force = np.zeros((len(gaussians), 3), dtype=np.float64)
    for pair in graph.active_contacts():
        left, right = penalty_force(pair, gaussians, mu=0.0)
        force[pair.i] += left
        force[pair.j] += right
    return force


def run_harmonic(directory: Path) -> dict:
    ply = _write_fixture(
        directory, "harmonic",
        xyz=[[1.0, 0.0, 0.0]],
        scale=[[0.05, 0.05, 0.05]],
        alpha=[0.5],
        velocity=[[0.0, 0.0, 0.0]],
        overrides={"rho": 1200.0, "E": 1.0e5, "nu": 0.3, "occupancy_amplitude": 1.0},
    )
    scene = compile_scene(ply)
    mass = float(scene.gaussians[0].mass)
    omega = float(np.sqrt(HARMONIC_K / mass))
    x0 = float(scene.gaussians[0].pos[0])
    rows = []
    for dt in HARMONIC_DTS:
        steps = int(round(HARMONIC_HORIZON / dt))
        state = State(
            pos=scene.gaussians[0].pos.copy()[None, :],
            vel=np.zeros((1, 3)),
        )
        samples = []
        kinetic = []
        potential = []

        def spring(pos, k=HARMONIC_K):
            return -k * pos

        for _ in range(steps):
            state = langevin_step(
                state, spring(state.pos), dt, np.array([mass]),
                T=0.0, gamma=0.0, force_fn=spring,
            )
            samples.append(float(state.pos[0, 0]))
            kinetic.append(0.5 * mass * float(np.sum(state.vel ** 2)))
            potential.append(0.5 * HARMONIC_K * float(np.sum(state.pos ** 2)))
        times = (np.arange(steps) + 1) * dt
        reference = x0 * np.cos(omega * times)
        mae = float(np.mean(np.abs(np.asarray(samples) - reference)))
        rows.append({
            "dt_s": dt,
            "steps": steps,
            "horizon_s": HORIZON_USED(steps, dt),
            "mae_m": mae,
            "energy_drift_per_s": float(eval_energy_drift(kinetic, potential, dt)),
            "energy_unit": "J",
            "position_unit": "m",
        })
    ratios = [
        rows[i]["mae_m"] / rows[i + 1]["mae_m"]
        for i in range(len(rows) - 1)
    ]
    return {
        "mass_kg": mass,
        "omega_rad_s": omega,
        "x0_m": x0,
        "spring_k_N_per_m": HARMONIC_K,
        "T_K": 0.0,
        "gamma_per_s": 0.0,
        "time_base": "sample i is the state after step i, at t=(i+1)*dt",
        "rows": rows,
        "mae_ratios": ratios,
    }


def HORIZON_USED(steps: int, dt: float) -> float:
    """Denominator of ``eval_energy_drift``: ``(len(E) - 1) * dt``."""
    return float((steps - 1) * dt)


def run_contact(directory: Path) -> dict:
    scale = 0.05
    ply = _write_fixture(
        directory, "contact",
        xyz=[[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]],
        scale=[[scale, scale, scale], [scale, scale, scale]],
        alpha=[0.5, 0.5],
        velocity=[[CONTACT_SPEED, 0.0, 0.0], [-CONTACT_SPEED, 0.0, 0.0]],
        overrides={"rho": 1200.0, "E": 1.0e5, "nu": 0.3, "occupancy_amplitude": 1.0},
    )
    scene = compile_scene(ply)
    gaussians = scene.gaussians
    state = State(
        pos=np.stack([gaussian.pos.copy() for gaussian in gaussians]),
        vel=scene.initial_velocities.copy(),
    )
    masses = np.array([gaussian.mass for gaussian in gaussians], dtype=np.float64)
    log = EventLog()
    graph = IncrementalContactGraph(threshold=0.0, log=log)
    for index, gaussian in enumerate(gaussians):
        graph.add(index, gaussian)
    graph.commit(gaussians, 0.0, state.vel)
    _append_spectrum(log, gaussians, 0.0, "initial")
    steps = int(round(CONTACT_HORIZON / CONTACT_DT))
    kinetic = []
    potential = []
    history = []
    mismatches = 0
    max_checked = 0
    for step in range(steps):
        pending = []

        def force_fn(pos, vel=state.vel.copy(), bucket=pending):
            for index in range(len(gaussians)):
                gaussians[index].pos = np.array(pos[index], dtype=np.float64, copy=True)
                if not np.array_equal(gaussians[index].pos, graph.bodies[index].pos):
                    graph.mark_moved(index)
            bucket.append(graph.commit(gaussians, (step + 1) * CONTACT_DT, vel))
            return _forces(graph, gaussians)

        state = langevin_step(
            state, _forces(graph, gaussians), CONTACT_DT, masses,
            T=0.0, gamma=0.0, force_fn=force_fn,
        )
        checked_this_step = len(graph.last_checked_pairs)
        for index in range(len(gaussians)):
            gaussians[index].pos = np.array(state.pos[index], dtype=np.float64, copy=True)
        graph.commit(gaussians, (step + 1) * CONTACT_DT, state.vel)
        if any(event["kind"] in ("contact_on", "contact_off") for event in pending[-1]):
            _append_spectrum(log, gaussians, (step + 1) * CONTACT_DT, "contact_topology")
        got = graph.active_contacts()
        ref = contact_pairs(gaussians, velocities=state.vel, threshold=0.0)
        if not _same_contacts(got, ref):
            mismatches += 1
        history.append(got)
        max_checked = max(max_checked, checked_this_step)
        kinetic.append(0.5 * float(np.sum(masses[:, None] * state.vel ** 2)))
        pot = 0.0
        for pair in got:
            normal_stiffness = float(
                pair.normal @ pairwise_stiffness(gaussians[pair.i], gaussians[pair.j]) @ pair.normal
            )
            pot += 0.5 * normal_stiffness * float(pair.penetration) ** 2
        potential.append(pot)
    drift = float(eval_energy_drift(kinetic, potential, CONTACT_DT))
    penetration = float(eval_penetration(history, gaussians))
    contact_steps = sum(1 for pairs in history if pairs)
    return {
        "scene": _scene_summary(scene),
        "dt_s": CONTACT_DT,
        "steps": steps,
        "horizon_s": HORIZON_USED(steps, CONTACT_DT),
        "requested_horizon_s": CONTACT_HORIZON,
        "T_K": 0.0,
        "gamma_per_s": 0.0,
        "friction_mu": 0.0,
        "activation_threshold": 0.0,
        "penetration_ratio": penetration,
        "penetration_limit": float(C.PEN_THRESHOLD),
        "penetration_definition": (
            "max over logged contacts of penetration / min(mean(scale_i), mean(scale_j)); "
            "eval_penetration; dimensionless"
        ),
        "energy_drift_per_s": drift,
        "energy_drift_limit_per_s": float(C.ENERGY_DRIFT_LIMIT),
        "energy_drift_definition": (
            "|E_end - E_0| / max(|E_0|, 1e-12) / ((n_samples - 1) * dt); "
            "E = kinetic + 0.5 * k_n * penetration^2; relative, unit 1/s"
        ),
        "contact_steps": int(contact_steps),
        "oracle_mismatches": int(mismatches),
        "narrowphase_checks": int(graph.narrowphase_checks),
        "max_checked_pairs_in_a_commit": int(max_checked),
        "full_rebuilds": int(graph.full_rebuilds),
        "events": log.events,
    }


def run_probe() -> dict:
    """One mover among separated bodies. Checks stay local; the oracle matches."""
    gaussians = []
    from occupancy_gaussian_framework import OccupancyGaussian
    for index in range(6):
        gaussians.append(OccupancyGaussian(
            pos=np.array([10.0 + 2.0 * index, 0.0, 0.0]),
            scale=np.ones(3) * 0.05,
            mass=1.0,
        ))
    gaussians.append(OccupancyGaussian(pos=np.array([0.0, 0.0, 0.0]), scale=np.ones(3) * 0.05, mass=1.0))
    gaussians.append(OccupancyGaussian(pos=np.array([0.08, 0.0, 0.0]), scale=np.ones(3) * 0.05, mass=1.0))
    graph = IncrementalContactGraph(threshold=0.0)
    for index, gaussian in enumerate(gaussians):
        graph.add(index, gaussian)
    graph.commit(gaussians, 0.0)
    before = graph.narrowphase_checks
    gaussians[6].pos = gaussians[6].pos + np.array([0.001, 0.0, 0.0])
    graph.mark_moved(6)
    graph.commit(gaussians, 0.01)
    got = graph.active_contacts()
    ref = oracle_all_pairs(gaussians, threshold=0.0)
    possible = len(gaussians) * (len(gaussians) - 1) // 2
    checked = list(graph.last_checked_pairs)
    return {
        "n_bodies": len(gaussians),
        "n_possible_pairs": possible,
        "checked_pairs_after_one_move": [list(pair) for pair in checked],
        "narrowphase_checks_for_that_commit": int(graph.narrowphase_checks - before),
        "contact_set_equal": _same_contacts(got, ref),
        "full_rebuilds_on_hot_path": int(graph.full_rebuilds),
        "untouched_pair_rechecked": any(pair != (6, 7) for pair in checked),
    }


def _scene_summary(scene) -> dict:
    return {
        "source_format": scene.metadata["source_format"],
        "representation": scene.metadata["representation"],
        "shape_dynamics": scene.metadata["shape_dynamics"],
        "n": len(scene.gaussians),
        "mass_kg": [float(gaussian.mass) for gaussian in scene.gaussians],
        "young_modulus_Pa": [float(value) for value in scene.young_moduli],
        "young_modulus_source": [row["young_modulus"]["source"] for row in scene.provenance],
        "alpha": [float(value) for value in scene.alphas],
        "occupancy_amplitude": [float(value) for value in scene.occupancy_amplitudes],
        "mass_density_kg_m3": [float(value) for value in scene.mass_densities],
        "length_unit": scene.metadata["length_unit"],
        "per_frame_3dgs_conversion": scene.metadata["per_frame_3dgs_conversion"],
    }


def _failures(harmonic, contact, probe) -> list[str]:
    failures = []
    ratios = harmonic["mae_ratios"]
    if any(ratio <= 3.0 for ratio in ratios):
        failures.append(f"harmonic MAE did not converge with ratio > 3: {ratios}")
    for row in harmonic["rows"]:
        if row["energy_drift_per_s"] >= C.ENERGY_DRIFT_LIMIT:
            failures.append(
                f"harmonic drift {row['energy_drift_per_s']} at dt={row['dt_s']} "
                f"is not below {C.ENERGY_DRIFT_LIMIT}/s"
            )
        if not np.isfinite(row["mae_m"]) or row["mae_m"] <= 0.0:
            failures.append(f"harmonic MAE is not a positive finite error: {row}")
    if contact["contact_steps"] <= 0:
        failures.append("contact scene never made contact; penetration would be vacuous")
    if contact["penetration_ratio"] <= 0.0:
        failures.append("penetration ratio is zero; the bodies did not overlap")
    if contact["penetration_ratio"] >= C.PEN_THRESHOLD:
        failures.append(
            f"penetration ratio {contact['penetration_ratio']} is not below "
            f"PEN_THRESHOLD {C.PEN_THRESHOLD}"
        )
    if contact["energy_drift_per_s"] >= C.ENERGY_DRIFT_LIMIT:
        failures.append(
            f"contact energy drift {contact['energy_drift_per_s']}/s "
            f"is not below {C.ENERGY_DRIFT_LIMIT}/s"
        )
    if contact["oracle_mismatches"] != 0:
        failures.append(f"incremental contacts diverged from all-pairs {contact['oracle_mismatches']} times")
    if contact["full_rebuilds"] != 0:
        failures.append("hot path performed a full rebuild")
    if contact["narrowphase_checks"] >= contact["steps"]:
        failures.append(
            "narrowphase ran on every step; separated frames should skip the predicate"
        )
    if not any(event["kind"] == "contact_on" for event in contact["events"]):
        failures.append("event log has no contact_on")
    if not any(event["kind"] == "contact_off" for event in contact["events"]):
        failures.append("event log has no contact_off")
    spectra = [event for event in contact["events"] if event["kind"] == "modal_spectrum"]
    topology = [event for event in contact["events"] if event["kind"] in ("contact_on", "contact_off")]
    initial = [event for event in spectra if event["reason"] == "initial"]
    followed = {event["time_s"] for event in spectra if event["reason"] == "contact_topology"}
    missing = {event["time_s"] for event in topology} - followed
    if len(initial) != 1 or missing:
        failures.append(
            f"spectrum log missing initial or a topology time; "
            f"missing={sorted(missing)} initial={len(initial)} spectra={len(spectra)}"
        )
    for event in spectra:
        freqs = event["frequencies_hz"]
        if not freqs or any(freq < -1e-9 or not np.isfinite(freq) for freq in freqs):
            failures.append(f"spectrum is not finite and non-negative: {freqs}")
        if any(b < a - 1e-9 for a, b in zip(freqs, freqs[1:])):
            failures.append(f"spectrum is not sorted: {freqs}")
        if event["mode_identity"] != "not_tracked":
            failures.append("M1 must not claim eigenmode tracking")
    seqs = [event["seq"] for event in contact["events"]]
    if seqs != list(range(len(seqs))):
        failures.append("event seq is not a chronological 0..n-1 replay key")
    if not probe["contact_set_equal"] or probe["full_rebuilds_on_hot_path"] != 0:
        failures.append(f"incremental probe failed oracle parity: {probe}")
    if probe["narrowphase_checks_for_that_commit"] != 1 or probe["untouched_pair_rechecked"]:
        failures.append(f"single-body update was not local: {probe}")
    return failures


def run_m1(artifact: Path = ARTIFACT) -> dict:
    import tempfile
    rng_before = C.RNG.bit_generator.state
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        harmonic = run_harmonic(folder)
        contact = run_contact(folder)
        contact_again = run_contact(folder)
    probe = run_probe()
    rng_after = C.RNG.bit_generator.state
    replay_equal = contact["events"] == contact_again["events"]
    failures = _failures(harmonic, contact, probe)
    if not replay_equal:
        failures.append("contact event log is not replay-stable")
    if rng_before != rng_after:
        failures.append("conservation runs consumed the global Langevin RNG")
    report = {
        "milestone": "M1",
        "seed": None,
        "determinism": "no RNG; T=0, gamma=0, friction_mu=0; event log replay matches",
        "rng_consumed": rng_before != rng_after,
        "event_replay_equal": replay_equal,
        "harmonic": harmonic,
        "contact": {key: value for key, value in contact.items() if key != "events"},
        "event_log": contact["events"],
        "incremental_probe": probe,
        "failures": failures,
        "passed": not failures,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    report = run_m1()
    print(json.dumps({
        "passed": report["passed"],
        "failures": report["failures"],
        "harmonic_mae_m": [row["mae_m"] for row in report["harmonic"]["rows"]],
        "harmonic_ratios": report["harmonic"]["mae_ratios"],
        "harmonic_drift_per_s": [row["energy_drift_per_s"] for row in report["harmonic"]["rows"]],
        "contact_penetration_ratio": report["contact"]["penetration_ratio"],
        "contact_drift_per_s": report["contact"]["energy_drift_per_s"],
        "contact_steps": report["contact"]["contact_steps"],
        "dt_s": report["contact"]["dt_s"],
        "horizon_s": report["contact"]["horizon_s"],
        "narrowphase_checks": report["contact"]["narrowphase_checks"],
        "probe_checks": report["incremental_probe"]["narrowphase_checks_for_that_commit"],
        "artifact": str(ARTIFACT),
    }, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
