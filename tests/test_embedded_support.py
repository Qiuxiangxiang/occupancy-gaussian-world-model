"""Embedded supports are frozen at compile time and are not contact events."""
from __future__ import annotations

import numpy as np

from compiler.scene_compiler import compile_scene
from occupancy_gaussian_framework import State, contact_pairs, langevin_step


def _scene(**metadata):
    return compile_scene({
        "xyz": np.array([[0.0, 0.0, -0.05], [0.0, 0.0, 0.4], [1.5, 0.0, 0.4]]),
        "scale": np.ones((3, 3)) * 0.1,
        "opacity": np.array([0.5, 0.5, 0.5]),
        "object_id": np.array([7, 7, 8]),
        "conventions": {"opacity": "alpha", "scale": "linear", "quaternion": "wxyz"},
        "metadata": {
            "gaussian_overrides": {
                "0": {"E": 1.0e5, "rho": 1200.0, "nu": 0.3, "occupancy_amplitude": 1.0},
                "1": {"E": 1.0e5, "rho": 1200.0, "nu": 0.3, "occupancy_amplitude": 1.0},
                "2": {"E": 1.0e5, "rho": 1200.0, "nu": 0.3, "occupancy_amplitude": 1.0},
            },
            **metadata,
        },
    })


def test_ground_plane_is_frozen_and_empty_contacts_do_not_create_it():
    scene = _scene(ground_plane={"normal": [0.0, 0.0, 1.0], "offset": 0.0})
    assert scene.supports.indices() == [0]
    assert scene.metadata["embedded_support"] == "frozen_at_compile"
    assert scene.provenance[0]["embedded_support"]["active_from_step"] == 0
    assert scene.provenance[1]["embedded_support"] is None
    state = State(
        pos=np.stack([gaussian.pos.copy() for gaussian in scene.gaussians]),
        vel=np.zeros((3, 3)),
    )
    assert contact_pairs(scene.gaussians, state.vel, threshold=0.0) == []
    displaced = state.pos.copy()
    displaced[0, 2] += 0.01
    force = scene.supports.force(displaced)
    energy = scene.supports.energy(displaced)
    stiffness = scene.supports.supports[0].stiffness_Npm
    assert abs(float(force[0, 2]) + stiffness * 0.01) < 1e-8
    assert abs(energy - 0.5 * stiffness * 0.01 ** 2) < 1e-8
    assert np.allclose(force[1], 0.0)
    assert scene.supports.evaluations == 1
    assert scene.supports.indices() == [0]


def test_explicit_index_and_default_stiffness_units():
    scene = _scene(embedded_indices=[1], support_stiffness=2500.0)
    support = scene.supports.supports[0]
    assert support.index == 1
    assert support.stiffness_Npm == 2500.0
    assert support.reasons == ("embedded_indices",)
    assert np.allclose(support.mask, 1.0)
    # Pa * m = N/m when the sidecar does not override k.
    defaulted = _scene(embedded_objects=[8])
    body = defaulted.gaussians[2]
    assert abs(defaulted.supports.supports[0].stiffness_Npm - body.E * float(np.mean(body.scale))) < 1e-8


def test_support_is_in_the_integrator_from_step_zero_without_contacts():
    scene = _scene(
        embedded_indices=[0],
        support_stiffness=4000.0,
        dof_mask=[0.0, 0.0, 1.0],
    )
    assert np.allclose(scene.supports.supports[0].mask, [0.0, 0.0, 1.0])
    gaussians = scene.gaussians
    state = State(
        pos=np.stack([gaussian.pos.copy() for gaussian in gaussians]),
        vel=np.zeros((3, 3)),
    )
    state.pos[0, 2] -= 0.002
    masses = np.array([gaussian.mass for gaussian in gaussians])
    seen = []
    for step in range(1000):
        pairs = contact_pairs(gaussians, state.vel, threshold=0.0)
        assert pairs == []
        assert scene.supports.indices() == [0]

        def force_fn(pos, pairs=pairs):
            assert pairs == []
            for index in range(len(gaussians)):
                gaussians[index].pos = pos[index]
            seen.append(scene.supports.indices())
            return scene.supports.force(pos)

        state = langevin_step(
            state, force_fn(state.pos), 1.0e-4, masses,
            T=0.0, gamma=0.0, force_fn=force_fn,
        )
    assert seen[0] == [0]
    assert all(item == [0] for item in seen)
    assert scene.supports.evaluations >= 21
    # The rooted center is pulled back toward the rest height. Lateral DOFs stay put.
    assert abs(float(state.pos[0, 2] - gaussians[0].pos[2])) < 1e-12
    rest_z = float(scene.supports.supports[0].rest_m[2])
    assert abs(float(state.pos[0, 2]) - (rest_z - 0.002)) > 1e-5
    assert np.allclose(state.pos[0, :2], scene.supports.supports[0].rest_m[:2])


def test_m1_scenes_do_not_grow_a_support():
    scene = compile_scene({
        "xyz": np.array([[-0.06, 0.0, 0.0], [0.06, 0.0, 0.0]]),
        "scale": np.ones((2, 3)) * 0.05,
        "opacity": np.array([0.5, 0.5]),
        "conventions": {"opacity": "alpha", "scale": "linear", "quaternion": "wxyz"},
    })
    assert scene.supports.indices() == []
    assert scene.metadata["embedded_support"] == "none"
    matrix = scene.supports.add_to_stiffness(np.eye(6))
    assert np.allclose(matrix, np.eye(6))


def main():
    test_ground_plane_is_frozen_and_empty_contacts_do_not_create_it()
    test_explicit_index_and_default_stiffness_units()
    test_support_is_in_the_integrator_from_step_zero_without_contacts()
    test_m1_scenes_do_not_grow_a_support()
    print("test_embedded_support ok")


if __name__ == "__main__":
    main()
