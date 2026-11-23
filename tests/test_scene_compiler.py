"""Compiler, Cross3DGS exchange, and material-guess tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np

from compiler.cross3dgs_ply import (
    CROSS3DGS_PLY_PROPERTIES,
    SH_C0,
    cloud_from_activated,
    read_cross3dgs_ply,
    write_cross3dgs_ply,
)
from compiler.errors import SceneCompileError
from compiler.scene_compiler import compile_scene
from occupancy_gaussian_framework import State


def _conventions(**extra):
    base = {"opacity": "logit", "scale": "log", "quaternion": "wxyz"}
    base.update(extra)
    return base


def test_ply_header_matches_save_gaussian_ply():
    cloud = cloud_from_activated(
        [[0.0, 0.0, 0.0]], [[0.2, 0.3, 0.4]], [0.4],
        [[1.0, 0.0, 0.0, 0.0]], [[0.1, 0.2, 0.3]],
    )
    rest = np.zeros((1, 45), dtype=np.float32)
    for channel in range(3):
        for coefficient in range(15):
            rest[0, channel * 15 + coefficient] = np.float32(0.01 * (channel * 15 + coefficient + 1))
    cloud.f_rest = rest
    quat = np.array([0.4, 0.3, -0.2, 0.8], dtype=np.float64)
    quat /= np.linalg.norm(quat)
    cloud.rot_wxyz = quat.astype(np.float32)[None, :]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "one.ply"
        write_cross3dgs_ply(path, cloud)
        text = path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
        props = [line[len("property float "):] for line in text.splitlines() if line.startswith("property float ")]
        assert props == list(CROSS3DGS_PLY_PROPERTIES)
        assert "nx" not in props
        assert len(props) == 59
        loaded = read_cross3dgs_ply(path)
    assert np.array_equal(loaded.f_rest, rest)
    assert np.array_equal(loaded.rot_wxyz, cloud.rot_wxyz)
    assert np.array_equal(loaded.opacity_logit, cloud.opacity_logit)
    assert np.array_equal(loaded.log_scale, cloud.log_scale)


def test_bit_exact_roundtrip_and_compile_activation():
    rng = np.random.default_rng(0)
    n = 3
    cloud = cloud_from_activated(
        rng.normal(size=(n, 3)),
        np.exp(rng.normal(scale=0.2, size=(n, 3))),
        np.full(n, 0.4),
        np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        rng.normal(size=(n, 3)),
        rng.normal(size=(n, 45)).astype(np.float32),
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cloud.ply"
        write_cross3dgs_ply(path, cloud)
        loaded = read_cross3dgs_ply(path)
        scene = compile_scene(path)
        write_cross3dgs_ply(Path(tmp) / "again.ply", scene.exchange)
        again = read_cross3dgs_ply(Path(tmp) / "again.ply")
    for name in ("xyz", "f_dc", "f_rest", "opacity_logit", "log_scale", "rot_wxyz"):
        assert np.array_equal(getattr(loaded, name), getattr(cloud, name))
        assert np.array_equal(getattr(again, name), getattr(cloud, name))
    assert scene.metadata["per_frame_3dgs_conversion"] is False
    assert scene.metadata["representation"] == "surface_contact"
    assert scene.metadata["shape_dynamics"] == "unsegmented_independent_particles"
    assert np.allclose(scene.alphas, 0.4, atol=1e-5)
    assert abs(scene.provenance[0]["occupancy_amplitude"]["value"] - (0.25 + 0.75 * 0.4)) < 1e-6


def test_logit_is_not_silently_read_as_alpha():
    common = dict(
        xyz=np.zeros((1, 3)),
        scale=np.log([[0.1, 0.1, 0.1]]),
        rotation=np.array([[1.0, 0.0, 0.0, 0.0]]),
        f_dc=np.zeros((1, 3)),
    )
    as_logit = compile_scene({**common, "opacity": np.array([0.5]), "conventions": _conventions()})
    as_alpha = compile_scene({
        **common, "opacity": np.array([0.5]),
        "conventions": _conventions(opacity="alpha", scale="log"),
    })
    assert abs(as_logit.alphas[0] - (1.0 / (1.0 + np.exp(-0.5)))) < 1e-8
    assert abs(as_alpha.alphas[0] - 0.5) < 1e-8
    assert abs(as_logit.alphas[0] - as_alpha.alphas[0]) > 0.05


def test_quaternion_order_and_marginal_scale():
    angle = np.pi / 2.0
    wxyz = np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    xyzw = np.array([0.0, 0.0, np.sin(angle / 2.0), np.cos(angle / 2.0)])
    scale = np.array([[2.0, 0.5, 0.5]])
    base = dict(xyz=np.array([[1.0, 0.0, 0.0]]), scale=scale, opacity=np.array([0.8]))
    a = compile_scene({**base, "rotation": wxyz[None, :], "conventions": _conventions(scale="linear", opacity="alpha")})
    b = compile_scene({
        **base, "rotation": xyzw[None, :],
        "conventions": _conventions(scale="linear", opacity="alpha", quaternion="xyzw"),
    })
    assert np.allclose(a.gaussians[0].scale, [0.5, 2.0, 0.5], atol=1e-5)
    assert np.allclose(a.gaussians[0].scale, b.gaussians[0].scale, atol=1e-5)


def test_validation_errors_name_the_field():
    def attempt(**kwargs):
        payload = dict(
            xyz=np.array([[0.0, 0.0, 0.0]]),
            scale=np.array([[0.1, 0.1, 0.1]]),
            opacity=np.array([0.5]),
            conventions=_conventions(scale="linear", opacity="alpha"),
        )
        payload.update(kwargs)
        compile_scene(payload)
    try:
        attempt(xyz=np.array([[np.nan, 0.0, 0.0]]))
        raise AssertionError("nan was accepted")
    except SceneCompileError as exc:
        assert "xyz" in str(exc)
    try:
        attempt(scale=np.array([[0.1, -0.2, 0.1]]))
        raise AssertionError("negative scale was accepted")
    except SceneCompileError as exc:
        assert "scale" in str(exc)
    try:
        attempt(metadata={"gaussian_overrides": {"0": {"nu": 0.5}}})
        raise AssertionError("nu=0.5 was accepted")
    except SceneCompileError as exc:
        assert "Poisson" in str(exc)
    try:
        attempt(metadata={"gaussian_overrides": {"0": {"E": -10.0}}})
        raise AssertionError("negative modulus was accepted")
    except SceneCompileError as exc:
        assert "Young" in str(exc)
    try:
        attempt(rotation=np.zeros((1, 4)))
        raise AssertionError("zero quaternion was accepted")
    except SceneCompileError as exc:
        assert "quaternion" in str(exc)
    try:
        attempt(metadata={"transform": np.diag([2.0, 1.0, 1.0, 1.0]).tolist()})
        raise AssertionError("scaled transform was accepted")
    except SceneCompileError as exc:
        assert "rigid" in str(exc)


def test_override_precedence_and_units():
    red = (np.array([0.80, 0.15, 0.15]) - 0.5) / SH_C0
    scene = compile_scene({
        "xyz": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "scale": np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]]),
        "opacity": np.array([0.2, 0.2]),
        "f_dc": np.vstack([red, red]),
        "object_id": np.array([3, 3]),
        "conventions": _conventions(scale="linear", opacity="alpha"),
        "metadata": {
            "object_overrides": {"3": {"material": "wood", "E": 3.0e5}},
            "gaussian_overrides": {"1": {"E": 4.0e4, "occupancy_amplitude": 0.5}},
        },
    })
    assert scene.provenance[0]["young_modulus"]["source"] == "object_override"
    assert scene.provenance[0]["young_modulus"]["value"] == 3.0e5
    assert scene.provenance[0]["rho"]["source"] == "object_material"
    assert scene.provenance[0]["rho"]["material"] == "wood"
    assert scene.provenance[1]["young_modulus"]["source"] == "gaussian_override"
    assert scene.provenance[1]["young_modulus"]["value"] == 4.0e4
    assert scene.provenance[1]["occupancy_amplitude"]["value"] == 0.5
    assert scene.alphas[1] != scene.occupancy_amplitudes[1]
    assert scene.mass_densities[1] != scene.young_moduli[1]
    assert "do not identify" in scene.metadata["disclaimer"]
    assert scene.metadata["shape_dynamics"].startswith("segmented_independent")
    assert scene.gaussians[0].tag == "obj:3"
    colored = compile_scene({
        "xyz": np.zeros((1, 3)),
        "scale": np.ones((1, 3)) * 0.1,
        "opacity": np.array([0.5]),
        "f_dc": red[None, :],
        "conventions": _conventions(scale="linear", opacity="alpha"),
    })
    assert colored.provenance[0]["young_modulus"]["source"] == "color_heuristic"
    assert colored.provenance[0]["young_modulus"]["material"] == "rubber"
    assert colored.provenance[0]["young_modulus"]["value"] == 2.0e5


def test_unit_scale_does_not_rewrite_exchange_bytes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "shift.ply"
        cloud = cloud_from_activated(
            [[100.0, 0.0, 0.0]], [[50.0, 50.0, 50.0]], [0.5],
            [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]],
        )
        write_cross3dgs_ply(path, cloud)
        (path.with_suffix(".json")).write_text(json.dumps({"unit_scale": 1.0}), encoding="utf-8")
        scaled = compile_scene(path, metadata={"unit_scale": 0.01, "length_unit": "m"})
    assert np.allclose(scaled.gaussians[0].pos, [1.0, 0.0, 0.0], atol=1e-5)
    assert np.allclose(scaled.gaussians[0].scale, [0.5, 0.5, 0.5], atol=1e-5)
    assert np.allclose(scaled.exchange.xyz[0], [100.0, 0.0, 0.0], atol=0.0)


def test_density_is_mass_kernel_not_alpha():
    scene = compile_scene({
        "xyz": np.zeros((1, 3)),
        "scale": np.ones((1, 3)) * 0.2,
        "opacity": np.array([0.25]),
        "conventions": _conventions(scale="linear", opacity="alpha"),
        "metadata": {"gaussian_overrides": {"0": {
            "rho": 1000.0, "E": 1.0e5, "nu": 0.3, "occupancy_amplitude": 0.4,
        }}},
    })
    peak = scene.density(np.zeros(3))
    assert abs(peak - 1000.0 * 0.4) / 400.0 < 1e-8
    assert abs(scene.alphas[0] - 0.25) < 1e-6
    pair = compile_scene({
        "xyz": np.zeros((2, 3)),
        "scale": np.ones((2, 3)) * 0.2,
        "opacity": np.array([0.25, 0.25]),
        "conventions": _conventions(scale="linear", opacity="alpha"),
        "metadata": {"gaussian_overrides": {
            "0": {"rho": 1000.0, "occupancy_amplitude": 0.4},
            "1": {"rho": 1000.0, "occupancy_amplitude": 0.4},
        }},
    })
    assert abs(pair.density(np.zeros(3)) - 2.0 * peak) / peak < 1e-6


def test_fixed_boundary_is_not_a_contact_product():
    scene = compile_scene({
        "xyz": np.array([[0.2, 0.0, 0.0]]),
        "scale": np.ones((1, 3)) * 0.1,
        "opacity": np.array([0.5]),
        "conventions": _conventions(scale="linear", opacity="alpha"),
        "metadata": {"gaussian_boundaries": {"0": {"kind": "fixed"}}},
    })
    state = State(pos=np.array([[5.0, 5.0, 5.0]]), vel=np.ones((1, 3)))
    scene.apply_boundary_conditions(state)
    assert np.allclose(state.pos, scene.gaussians[0].pos)
    assert np.allclose(state.vel, 0.0)
    assert scene.boundary_conditions[0].source == "gaussian_override"


def test_empty_cloud():
    scene = compile_scene({
        "xyz": np.zeros((0, 3)),
        "scale": np.zeros((0, 3)),
        "opacity": np.zeros((0,)),
        "conventions": _conventions(scale="linear", opacity="alpha"),
    })
    assert scene.gaussians == []
    assert scene.density([1.0, 2.0, 3.0]) == 0.0


def main():
    test_ply_header_matches_save_gaussian_ply()
    test_bit_exact_roundtrip_and_compile_activation()
    test_logit_is_not_silently_read_as_alpha()
    test_quaternion_order_and_marginal_scale()
    test_validation_errors_name_the_field()
    test_override_precedence_and_units()
    test_unit_scale_does_not_rewrite_exchange_bytes()
    test_density_is_mass_kernel_not_alpha()
    test_fixed_boundary_is_not_a_contact_product()
    test_empty_cloud()
    print("test_scene_compiler ok")


if __name__ == "__main__":
    main()
