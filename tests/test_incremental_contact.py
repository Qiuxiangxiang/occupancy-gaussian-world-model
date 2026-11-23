"""Incremental contact graph against the all-pairs oracle."""
from __future__ import annotations

import numpy as np

from compiler.contact_graph import (
    IncrementalContactGraph,
    oracle_all_pairs,
    oracle_full_rebuild,
)
from occupancy_gaussian_framework import OccupancyGaussian


def _body(pos, scale):
    return OccupancyGaussian(
        pos=np.array(pos, dtype=np.float64),
        scale=np.array(scale, dtype=np.float64),
        mass=1.0,
    )


def _keys(pairs):
    return [(int(pair.i), int(pair.j)) for pair in pairs]


def _assert_match(gaussians, graph, threshold, velocities=None):
    got = graph.active_contacts()
    ref = oracle_all_pairs(gaussians, velocities=velocities, threshold=threshold)
    assert _keys(got) == _keys(ref)
    for left, right in zip(got, ref):
        assert abs(float(left.penetration) - float(right.penetration)) < 1e-8
        assert np.linalg.norm(left.normal - right.normal) < 1e-8
    rebuilt = oracle_full_rebuild(gaussians, threshold=threshold, velocities=velocities)
    assert graph.full_rebuilds == 0
    assert rebuilt.full_rebuilds == 1
    assert _keys(rebuilt.active_contacts()) == _keys(ref)


def test_empty_single_and_dirty_read():
    graph = IncrementalContactGraph()
    assert graph.commit([], 0.0) == []
    assert graph.active_contacts() == []
    graph.add(0, _body([0, 0, 0], [0.2, 0.2, 0.2]))
    try:
        graph.active_contacts()
        raise AssertionError("dirty graph returned contacts")
    except RuntimeError as exc:
        assert "dirty" in str(exc)
    bodies = [_body([0, 0, 0], [0.2, 0.2, 0.2])]
    graph.commit(bodies, 0.0)
    assert graph.active_contacts() == []
    try:
        IncrementalContactGraph(support_factor=0.5)
        raise AssertionError("unsafe support factor was accepted")
    except ValueError:
        pass


def test_initial_overlap_is_an_event_without_motion():
    bodies = [
        _body([-0.05, 0, 0], [0.2, 0.2, 0.2]),
        _body([0.05, 0, 0], [0.2, 0.2, 0.2]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    graph.add(0, bodies[0])
    graph.add(1, bodies[1])
    events = graph.commit(bodies, 0.0)
    assert [event["kind"] for event in events] == ["contact_on"]
    assert events[0]["pair_id"] == "0-1"
    _assert_match(bodies, graph, 0.0)
    checks = graph.narrowphase_checks
    assert graph.commit(bodies, 0.1) == []
    assert graph.narrowphase_checks == checks


def test_local_update_does_not_recheck_an_unrelated_pair():
    bodies = [
        _body([0.0, 0, 0], [0.2, 0.2, 0.2]),
        _body([0.1, 0, 0], [0.2, 0.2, 0.2]),
        _body([5.0, 0, 0], [0.1, 0.1, 0.1]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    assert _keys(graph.active_contacts()) == [(0, 1)]
    bodies[2].pos = np.array([5.001, 0.0, 0.0])
    graph.mark_moved(2)
    events = graph.commit(bodies, 0.2)
    assert events == []
    assert (0, 1) not in graph.last_checked_pairs
    assert _keys(graph.active_contacts()) == [(0, 1)]
    bodies[2].scale = np.ones(3) * 0.05
    graph.mark_moved(2)
    graph.commit(bodies, 0.3)
    assert (0, 1) not in graph.last_checked_pairs


def test_scale_and_negative_coordinates_match_oracle():
    bodies = [
        _body([-5.0, -1.0, 0.2], [0.1, 0.1, 0.1]),
        _body([-4.7, -1.0, 0.2], [0.1, 0.1, 0.1]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    assert graph.active_contacts() == []
    bodies[1].scale = np.ones(3) * 0.25
    graph.mark_moved(1)
    events = graph.commit(bodies, 1.0)
    assert any(event["kind"] == "contact_on" for event in events)
    _assert_match(bodies, graph, 0.0)


def test_anisotropic_narrowphase_rejects_a_broadphase_neighbor():
    bodies = [
        _body([0, 0, 0], [1.0, 0.01, 0.01]),
        _body([0, 0.05, 0], [1.0, 0.01, 0.01]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    assert graph.narrowphase_checks == 1
    assert graph.active_contacts() == []
    _assert_match(bodies, graph, 0.0)


def test_negative_threshold_still_finds_the_widened_contact():
    threshold = -0.1
    bodies = [
        _body([0, 0, 0], [0.2, 0.2, 0.2]),
        _body([0.41, 0, 0], [0.2, 0.2, 0.2]),
    ]
    graph = IncrementalContactGraph(threshold=threshold)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    _assert_match(bodies, graph, threshold)
    assert _keys(graph.active_contacts()) == [(0, 1)]


def test_boundary_and_heterogeneous_scales():
    inside = [
        _body([-2.0, 0, 0], [0.2, 0.2, 0.2]),
        _body([-2.0 + 0.4 - 1e-6, 0, 0], [0.2, 0.2, 0.2]),
    ]
    outside = [
        _body([3.0, -4.0, 0], [0.2, 0.2, 0.2]),
        _body([3.0 + 0.4 + 1e-4, -4.0, 0], [0.2, 0.2, 0.2]),
    ]
    mixed = [
        _body([0, 0, 0], [5.0, 5.0, 5.0]),
        _body([9.0, 0, 0], [0.01, 0.01, 0.01]),
        _body([30.0, 0, 0], [0.01, 0.01, 0.01]),
    ]
    for bodies in (inside, outside, mixed):
        graph = IncrementalContactGraph(threshold=0.0)
        for index, body in enumerate(bodies):
            graph.add(index, body)
        graph.commit(bodies, 0.0)
        _assert_match(bodies, graph, 0.0)
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(inside):
        graph.add(index, body)
    graph.commit(inside, 0.0)
    assert len(graph.active_contacts()) == 1
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(outside):
        graph.add(index, body)
    graph.commit(outside, 0.0)
    assert graph.active_contacts() == []


def test_motion_safety_and_tunnel_do_not_invent_an_endpoint_contact():
    bodies = [
        _body([0.0, 0, 0], [0.2, 0.2, 0.2]),
        _body([5.0, 0, 0], [0.2, 0.2, 0.2]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    bodies[0].pos = np.array([10.0, 0.0, 0.0])
    graph.mark_moved(0)
    events = graph.commit(bodies, 1.0)
    kinds = [event["kind"] for event in events]
    assert "motion_safety" in kinds
    assert "possible_tunnel" in kinds
    assert "contact_on" not in kinds
    assert graph.active_contacts() == []
    assert graph.full_rebuilds == 0
    _assert_match(bodies, graph, 0.0)


def test_remove_and_velocity_refresh():
    bodies = [
        _body([0, 0, 0], [0.2, 0.2, 0.2]),
        _body([0.05, 0, 0], [0.2, 0.2, 0.2]),
        _body([4.0, 0, 0], [0.1, 0.1, 0.1]),
    ]
    graph = IncrementalContactGraph(threshold=0.0)
    for index, body in enumerate(bodies):
        graph.add(index, body)
    graph.commit(bodies, 0.0)
    checks = graph.narrowphase_checks
    vel = np.zeros((3, 3))
    vel[1, 0] = 1.5
    events = graph.commit(bodies, 0.2, vel)
    assert events == []
    assert graph.narrowphase_checks == checks
    assert abs(float(graph.active_contacts()[0].rel_vel[0]) - 1.5) < 1e-12
    removed = graph.remove(0, 0.3)
    assert [event["kind"] for event in removed] == ["contact_off"]
    assert graph.active_contacts() == []
    assert graph.full_rebuilds == 0


def test_random_moves_match_oracle():
    rng = np.random.default_rng(20261122)
    for trial in range(12):
        n = int(rng.integers(1, 9))
        pos = rng.normal(loc=-1.5, scale=1.2, size=(n, 3))
        scale = np.exp(rng.uniform(np.log(1e-3), np.log(0.8), size=(n, 3)))
        if trial == 0 and n >= 2:
            scale[0] = np.array([2.0, 0.01, 0.01])
            scale[1] = np.array([1e-3, 1e-3, 1e-3])
        threshold = float(rng.choice([-0.2, 0.0, 0.05]))
        bodies = [_body(pos[i], scale[i]) for i in range(n)]
        graph = IncrementalContactGraph(threshold=threshold)
        for index, body in enumerate(bodies):
            graph.add(index, body)
        graph.commit(bodies, 0.0)
        _assert_match(bodies, graph, threshold)
        mover = int(rng.integers(0, n))
        bodies[mover].pos = bodies[mover].pos + rng.normal(scale=0.4, size=3)
        graph.mark_moved(mover)
        graph.commit(bodies, 0.1)
        _assert_match(bodies, graph, threshold)
        bodies[mover].scale = bodies[mover].scale * float(rng.uniform(0.5, 2.0))
        graph.mark_moved(mover)
        graph.commit(bodies, 0.2)
        _assert_match(bodies, graph, threshold)
        if n > 1:
            victim = n - 1
            graph.remove(victim, 0.3)
            got = _keys(graph.active_contacts())
            ref = [
                key for key in _keys(oracle_all_pairs(bodies, threshold=threshold))
                if victim not in key
            ]
            assert got == ref
            assert graph.full_rebuilds == 0


def main():
    test_empty_single_and_dirty_read()
    test_initial_overlap_is_an_event_without_motion()
    test_local_update_does_not_recheck_an_unrelated_pair()
    test_scale_and_negative_coordinates_match_oracle()
    test_anisotropic_narrowphase_rejects_a_broadphase_neighbor()
    test_negative_threshold_still_finds_the_widened_contact()
    test_boundary_and_heterogeneous_scales()
    test_motion_safety_and_tunnel_do_not_invent_an_endpoint_contact()
    test_remove_and_velocity_refresh()
    test_random_moves_match_oracle()
    print("test_incremental_contact ok")


if __name__ == "__main__":
    main()
