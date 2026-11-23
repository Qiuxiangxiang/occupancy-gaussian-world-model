"""M3 symbolic switches: hysteresis, three axes, replay, no full rebuild."""
from __future__ import annotations

import numpy as np

from compiler.contact_graph import IncrementalContactGraph
from compiler.events import EventLog
from compiler.topology import (
    PEN_RATIO_ENTER,
    SymbolicTopology,
    modal_participation,
    observe_graph,
    pair_geometry,
)
from occupancy_gaussian_framework import OccupancyGaussian, modal_analysis


def _body(pos, scale=0.2, object_id=None, nu=0.3):
    gaussian = OccupancyGaussian(
        pos=np.array(pos, dtype=np.float64),
        scale=np.ones(3) * scale,
        mass=1.0,
        tag="unsegmented" if object_id is None else f"obj:{object_id}",
        nu=nu,
    )
    return gaussian


def _at_ratio(ratio, scale=0.2):
    """Two equal spheres whose penetration ratio is about ``ratio``."""
    distance = scale * (2.0 - ratio)
    return [_body([0.0, 0.0, 0.0], scale), _body([distance, 0.0, 0.0], scale)]


def _drive(ratios, object_ids=(1, 2), mu=0.0, speed=0.0, modes=None):
    log = EventLog()
    topology = SymbolicTopology(log, object_ids, mu=mu)
    for step, ratio in enumerate(ratios):
        bodies = _at_ratio(ratio)
        velocities = np.zeros((2, 3))
        velocities[1, 1] = speed
        topology.observe(0.01 * step, bodies, _as_contacts(bodies, velocities), velocities, modes)
    return log


def _as_contacts(bodies, velocities):
    from occupancy_gaussian_framework import contact_between
    pair = contact_between(bodies, 0, 1, velocities, threshold=0.0)
    return [] if pair is None else [pair]


def test_hysteresis_ignores_chatter_below_the_enter_band():
    log = _drive([0.0, 0.01, 0.0, 0.015, 0.0])
    assert log.events == []
    entered = _drive([0.0, 0.01, 0.05, 0.01, 0.015, -0.05])
    normals = [event["after"] for event in entered.events if event["axis"] == "normal"]
    assert normals == ["point", "separated"]
    assert entered.events[0]["pen_ratio"] > PEN_RATIO_ENTER
    assert all(event["volumetric_incompressibility"] is False for event in entered.events)
    assert all(event["physics"] != "complementarity" for event in entered.events)


def test_point_area_and_self_are_different_labels():
    point = _drive([0.05], object_ids=(1, 2))
    assert [event["after"] for event in point.events if event["axis"] == "normal"] == ["point"]
    deep = _drive([0.05, 0.25, 0.15, 0.05], object_ids=(1, 2))
    assert [event["after"] for event in deep.events if event["axis"] == "normal"] == [
        "point", "area", "point",
    ]
    self_pen = _drive([0.05], object_ids=(4, 4))
    assert [event["after"] for event in self_pen.events if event["axis"] == "normal"] == [
        "self_penetration",
    ]
    assert self_pen.events[0]["physics"] == "penalty_contact"


def test_two_pairs_between_the_same_objects_are_area():
    scale = 0.2
    distance = scale * (2.0 - 0.05)
    bodies = [
        _body([0.0, 0.0, 0.0], scale),
        _body([distance, 0.0, 0.0], scale),
        _body([0.0, distance, 0.0], scale),
    ]
    from occupancy_gaussian_framework import contact_between
    contacts = [
        contact_between(bodies, 0, 1, threshold=0.0),
        contact_between(bodies, 0, 2, threshold=0.0),
    ]
    log = EventLog()
    SymbolicTopology(log, [1, 2, 2], mu=0.0).observe(0.0, bodies, contacts)
    labels = {
        event["pair_id"]: event["after"]
        for event in log.events if event["axis"] == "normal"
    }
    assert labels == {"0-1": "area", "0-2": "area"}


def test_mu0_is_free_not_sticking_or_sliding():
    log = _drive([0.05], object_ids=(1, 2), mu=0.0, speed=1.0)
    tangential = [event for event in log.events if event["axis"] == "tangential"]
    assert [event["after"] for event in tangential] == ["free"]
    assert tangential[0]["physics"] == "no_tangential_constraint"
    assert "sticking" not in [event["after"] for event in log.events]
    assert "sliding" not in [event["after"] for event in log.events]


def test_coulomb_stick_slide_hysteresis():
    log = EventLog()
    topology = SymbolicTopology(log, [1, 2], mu=0.3)
    sequence = [
        (0.05, 0.0),
        (0.05, 5.0e-3),
        (0.05, 1.5e-3),
        (0.05, 0.0),
    ]
    for step, (ratio, speed) in enumerate(sequence):
        bodies = _at_ratio(ratio)
        velocities = np.zeros((2, 3))
        velocities[1, 1] = speed
        topology.observe(0.1 * step, bodies, _as_contacts(bodies, velocities), velocities)
    tangential = [event["after"] for event in log.events if event["axis"] == "tangential"]
    assert tangential == ["sticking", "sliding", "sticking"]


def test_participation_threshold_skips_rigid_modes_and_small_scores():
    mass = np.eye(2)
    stiffness = np.diag([(2.0 * np.pi * 0.2) ** 2, (2.0 * np.pi * 10.0) ** 2])
    freqs, vectors = modal_analysis(stiffness, mass, n_modes=2)
    low = modal_participation(vectors[:, 0], vectors, freqs)
    assert low["excited_tracked_ids"] == []
    high = modal_participation(vectors[:, 1], vectors, freqs)
    assert high["excited_tracked_ids"] == [1]
    assert high["scores"][1]["score"] > 0.99


def test_switch_carries_the_same_participation_as_the_penalty_force():
    bodies = _at_ratio(0.05)
    velocities = np.zeros((2, 3))
    vectors = np.eye(6)
    modes = {"vectors": vectors, "frequencies_hz": [0.0, 0.0, 0.0, 10.0, 20.0, 30.0]}
    log = EventLog()
    SymbolicTopology(log, [1, 2], mu=0.0).observe(
        0.0, bodies, _as_contacts(bodies, velocities), velocities, modes,
    )
    modal = [event for event in log.events if event["axis"] == "modal_participation"]
    assert len(modal) == 1
    assert modal[0]["before"] == []
    assert modal[0]["physics"] == "modal_force_fraction"
    assert abs(sum(item["score"] for item in modal[0]["scores"]) - 1.0) < 1e-8


def test_replay_is_identical_and_support_is_announced_once():
    def run():
        log = EventLog()
        topology = SymbolicTopology(log, [1, 2], mu=0.0, embedded_indices=[0])
        for step, ratio in enumerate([0.0, 0.05, 0.01, -0.05]):
            bodies = _at_ratio(ratio)
            velocities = np.zeros((2, 3))
            topology.observe(0.2 * step, bodies, _as_contacts(bodies, velocities), velocities)
        return log.events
    first, second = run(), run()
    assert first == second
    supports = [event for event in first if event["axis"] == "embedded_support"]
    assert len(supports) == 1
    assert supports[0]["before"] == "inactive" and supports[0]["after"] == "active"
    assert supports[0]["physics"] == "embedded_quadratic"
    assert first[0]["axis"] == "embedded_support"


def test_graph_observer_does_not_rebuild_and_nu_is_not_a_switch():
    bodies = _at_ratio(0.05)
    for body in bodies:
        body.nu = 0.49
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    log = EventLog()
    topology = SymbolicTopology(log, [None, None], mu=0.0)
    observe_graph(topology, graph, 0.0, bodies)
    assert graph.full_rebuilds == 0
    assert topology.full_rebuilds == 0
    assert any(event["axis"] == "normal" and event["after"] == "point" for event in log.events)
    assert all(event["volumetric_incompressibility"] is False for event in log.events)
    ratio = pair_geometry(bodies, 0, 1)["pen_ratio"]
    assert 0.02 < ratio < 0.20


def main():
    test_hysteresis_ignores_chatter_below_the_enter_band()
    test_point_area_and_self_are_different_labels()
    test_two_pairs_between_the_same_objects_are_area()
    test_mu0_is_free_not_sticking_or_sliding()
    test_coulomb_stick_slide_hysteresis()
    test_participation_threshold_skips_rigid_modes_and_small_scores()
    test_switch_carries_the_same_participation_as_the_penalty_force()
    test_replay_is_identical_and_support_is_announced_once()
    test_graph_observer_does_not_rebuild_and_nu_is_not_a_switch()
    print("test_topology ok")


if __name__ == "__main__":
    main()
