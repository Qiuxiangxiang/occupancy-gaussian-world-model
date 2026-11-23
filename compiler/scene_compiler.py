"""Compile a Cross3DGS point cloud into surface-contact initial conditions.

Each imported splat becomes one occupancy gaussian: a surface element, not a
filled volume and not a rigid body. Object ids, when a sidecar provides them,
select materials and labels only. Rigid reduction and embedded supports are
not applied.

World numbers stay in the file's coordinates unless ``unit_scale`` or a rigid
``transform`` is set. Those affect the physics state. The exchange payload
keeps the file-domain floats so a later ``write_cross3dgs_ply`` is an export,
not a per-frame conversion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from compiler.cross3dgs_ply import (
    SH_C0,
    Cross3DGSCloud,
    cloud_from_activated,
    read_cross3dgs_ply,
)
from compiler.errors import SceneCompileError
from compiler.materials import (
    DISCLAIMER,
    amplitude_from_alpha,
    default_mapping_table,
    resolve_materials,
    sh_dc_to_rgb,
)
from occupancy_gaussian_framework import OccupancyGaussian


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def _marginal_std(principal: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rot = _quat_to_matrix(quat_wxyz)
    # Diagonal of R diag(s^2) R^T is (R^2) @ s^2.
    variance = (rot ** 2) @ (principal ** 2)
    return np.sqrt(np.maximum(variance, 0.0))


def _as_metadata(value) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise SceneCompileError("metadata must be a dict")
    return value


def _load_sidecar(path: Path) -> dict:
    sidecar = path.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    with sidecar.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise SceneCompileError(f"{sidecar} must contain a JSON object")
    return data


def _merge_metadata(base: dict, override: dict) -> dict:
    merged = dict(base)
    merged.update(override)
    return merged


@dataclass
class BoundaryCondition:
    index: int
    kind: str
    source: str
    anchor: np.ndarray | None = None
    velocity: np.ndarray | None = None


@dataclass
class CompiledScene:
    """Physics initial condition plus the untouched exchange payload.

    ``gaussians`` use diagonal world-axis scales because Phase 0 kernels are
    diagonal. ``exchange`` still holds the Cross3DGS file-domain splat.
    """

    gaussians: list[OccupancyGaussian]
    boundary_conditions: list[BoundaryCondition]
    provenance: list[dict]
    exchange: Cross3DGSCloud
    initial_velocities: np.ndarray
    alphas: np.ndarray
    occupancy_amplitudes: np.ndarray
    mass_densities: np.ndarray
    young_moduli: np.ndarray
    metadata: dict = field(default_factory=dict)

    def density(self, x) -> float:
        """Mass density (kg/m^3 if the caller used meters) of the lumped kernels.

        Each kernel integrates to that gaussian's mass. Overlaps add. This is
        not visual alpha, not the occupancy amplitude, and not Young's modulus.
        """
        point = np.asarray(x, dtype=np.float64).reshape(3)
        total = 0.0
        for gaussian in self.gaussians:
            delta = point - gaussian.pos
            quad = 0.5 * float(delta @ gaussian.cov_inv @ delta)
            norm = (2.0 * np.pi) ** -1.5 / float(np.prod(gaussian.scale))
            total += gaussian.mass * norm * np.exp(-quad)
        return float(total)

    def apply_boundary_conditions(self, state) -> None:
        """Hold fixed anchors and overwrite prescribed velocities.

        A fixed boundary is a Dirichlet condition the sidecar named. It is not
        an embedded volume support and it is not created by a contact event.
        """
        for bc in self.boundary_conditions:
            if bc.kind == "free":
                continue
            if bc.kind == "fixed":
                state.pos[bc.index] = np.array(bc.anchor, dtype=np.float64, copy=True)
                state.vel[bc.index] = 0.0
            elif bc.kind == "prescribed_velocity":
                state.vel[bc.index] = np.array(bc.velocity, dtype=np.float64, copy=True)
            else:
                raise SceneCompileError(f"unknown boundary {bc.kind}")


def compile_scene(source, metadata: dict | None = None, mapping: dict | None = None) -> CompiledScene:
    """Compile a ``.ply`` path, a ``.npz`` path, or an in-memory dict.

    PLY input uses the Cross3DGS file-domain conventions and will not reinterpret
    logits as alpha. Dict and npz input must set ``conventions`` explicitly.
    """
    override = _as_metadata(metadata)
    if isinstance(source, Cross3DGSCloud):
        cloud = source
        meta = override
        source_format = "cross3dgs_cloud"
        present_rest = True
        present_dc = True
    elif isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".ply":
            cloud = read_cross3dgs_ply(path)
            meta = _merge_metadata(_load_sidecar(path), override)
            source_format = "cross3dgs_ply"
            present_rest = bool(getattr(cloud, "file_has_f_rest", False))
            present_dc = bool(getattr(cloud, "file_has_f_dc", False))
        elif suffix == ".npz":
            cloud, meta_file, flags = _cloud_from_npz(path)
            meta = _merge_metadata(meta_file, override)
            source_format = "internal_npz"
            present_rest = flags["f_rest"]
            present_dc = flags["f_dc"]
        else:
            raise SceneCompileError(
                f"unsupported file {path.name}; use a Cross3DGS .ply or an internal .npz"
            )
    elif isinstance(source, dict):
        cloud, flags = _cloud_from_mapping(source)
        meta = _merge_metadata(source.get("metadata") or {}, override)
        for key in ("object_id", "initial_velocity"):
            if key in source and key not in meta:
                meta[key] = source[key]
        source_format = "internal_dict"
        present_rest = flags["f_rest"]
        present_dc = flags["f_dc"]
    else:
        raise SceneCompileError(f"cannot compile {type(source).__name__}")

    return _compile_cloud(
        cloud, meta, mapping or default_mapping_table(), source_format,
        present_dc, present_rest,
    )


def _compile_cloud(cloud: Cross3DGSCloud, meta: dict, table: dict,
                   source_format: str, present_dc: bool, present_rest: bool) -> CompiledScene:
    n = cloud.count
    unit_scale = float(meta.get("unit_scale", 1.0))
    if not np.isfinite(unit_scale) or unit_scale <= 0.0:
        raise SceneCompileError("unit_scale must be finite and positive")
    transform = _rigid_transform(meta.get("transform"))
    object_ids = _object_ids(n, meta.get("object_id"))
    logits = np.asarray(cloud.opacity_logit, dtype=np.float64)
    log_scale = np.asarray(cloud.log_scale, dtype=np.float64)
    xyz = np.asarray(cloud.xyz, dtype=np.float64)
    quats = np.asarray(cloud.rot_wxyz, dtype=np.float64)
    f_dc = np.asarray(cloud.f_dc, dtype=np.float64)
    _require_finite("xyz", xyz)
    _require_finite("opacity logit", logits)
    _require_finite("log scale", log_scale)
    _require_finite("quaternion", quats)
    _require_finite("f_dc", f_dc)
    alpha = _sigmoid(logits)
    principal = np.exp(log_scale) * unit_scale
    if np.any(principal <= 0.0) or np.any(~np.isfinite(principal)):
        raise SceneCompileError("log scale produced a non-positive or non-finite principal scale")
    positions = xyz * unit_scale
    world_quats = np.zeros_like(quats)
    world_scales = np.zeros_like(principal)
    quat_renormalized = []
    for i in range(n):
        quat, renamed = _normalize_quat(quats[i], i)
        if transform is not None:
            quat = _quat_mul(transform["quat"], quat)
            quat, _ = _normalize_quat(quat, i)
            positions[i] = transform["rot"] @ positions[i] + transform["t"]
        rot = _quat_to_matrix(quat)
        det = float(np.linalg.det(rot))
        if abs(det - 1.0) > 1e-5:
            raise SceneCompileError(
                f"gaussian {i} quaternion is not a proper rotation (det={det})"
            )
        world_quats[i] = quat
        world_scales[i] = _marginal_std(principal[i], quat)
        quat_renormalized.append(renamed)
    if np.any(world_scales <= 0.0) or np.any(~np.isfinite(world_scales)):
        raise SceneCompileError("world-axis scale is not finite and positive")
    if np.any(~np.isfinite(positions)):
        raise SceneCompileError("position is not finite after unit_scale and transform")

    rgb = sh_dc_to_rgb(f_dc) if present_dc else None
    resolved = resolve_materials(n, object_ids, rgb, meta, table)
    gaussians = []
    boundaries = []
    provenance = []
    amplitudes = np.zeros(n)
    rhos = np.zeros(n)
    young = np.zeros(n)
    for i in range(n):
        row = resolved[i]
        if row["amplitude"] is None:
            row["amplitude"] = amplitude_from_alpha(float(alpha[i]), row["amplitude_needs_alpha"])
        _validate_physics(i, row, world_scales[i])
        volume = float((2.0 * np.pi) ** 1.5 * np.prod(world_scales[i]))
        amp = float(row["amplitude"]["value"])
        rho = float(row["rho"]["value"])
        mass = rho * amp * volume
        if not np.isfinite(mass) or mass <= 0.0:
            raise SceneCompileError(f"gaussian {i} mass {mass} is not finite and positive")
        segmented = object_ids[i] is not None
        tag = f"obj:{object_ids[i]}" if segmented else "unsegmented"
        gaussian = OccupancyGaussian(
            pos=positions[i].copy(),
            scale=world_scales[i].copy(),
            mass=float(mass),
            tag=tag,
            E=float(row["E"]["value"]),
            nu=float(row["nu"]["value"]),
            rho=rho,
        )
        gaussians.append(gaussian)
        boundary = row["boundary"]
        anchor = positions[i].copy() if boundary["kind"] == "fixed" else None
        velocity = None if boundary["velocity"] is None else np.asarray(boundary["velocity"], dtype=np.float64)
        boundaries.append(BoundaryCondition(
            index=i, kind=boundary["kind"], source=boundary["source"],
            anchor=anchor, velocity=velocity,
        ))
        amplitudes[i] = amp
        rhos[i] = rho
        young[i] = float(row["E"]["value"])
        provenance.append({
            "index": i,
            "object_id": object_ids[i],
            "segmented": segmented,
            "alpha": float(alpha[i]),
            "rgb": None if rgb is None else rgb[i].tolist(),
            "occupancy_amplitude": row["amplitude"],
            "rho": row["rho"],
            "young_modulus": row["E"],
            "nu": row["nu"],
            "friction_mu": row["mu"],
            "boundary": {"kind": boundary["kind"], "source": boundary["source"]},
            "principal_scale_m": principal[i].tolist(),
            "world_scale_m": world_scales[i].tolist(),
            "effective_volume_m3": volume,
            "mass_kg": float(mass),
            "peak_mass_density": float(rho * amp),
            "quaternion_wxyz": world_quats[i].tolist(),
            "quaternion_renormalized": bool(quat_renormalized[i]),
            "scale_note": (
                "Phase 0 stores the diagonal world-axis marginal std of the "
                "rotated Cross3DGS covariance. Dynamics do not use the full "
                "oriented ellipsoid. Volume uses that diagonal kernel, which "
                "matches the principal volume only when the splat is axis-aligned."
            ),
        })
    velocities = _initial_velocities(n, meta.get("initial_velocity"))
    any_segmented = any(oid is not None for oid in object_ids)
    scene_meta = {
        "source_format": source_format,
        "representation": "surface_contact",
        "volume_embedding": "deferred_m2",
        "rigid_reduction": "deferred",
        "shape_dynamics": (
            "segmented_independent_particles_rigid_reduction_deferred"
            if any_segmented else "unsegmented_independent_particles"
        ),
        "per_frame_3dgs_conversion": False,
        "length_unit": meta.get("length_unit", "file"),
        "unit_scale": unit_scale,
        "disclaimer": DISCLAIMER,
        "sh_c0": SH_C0,
        "appearance_present": {"f_dc": present_dc, "f_rest": present_rest},
        "exchange_schema": "cross3dgs-ply-SaveGaussianPly",
        "exchange_commit": "d69f79e1431d39d7ba71e22c29065dca67a4b4ce",
    }
    return CompiledScene(
        gaussians=gaussians,
        boundary_conditions=boundaries,
        provenance=provenance,
        exchange=cloud,
        initial_velocities=velocities,
        alphas=alpha,
        occupancy_amplitudes=amplitudes,
        mass_densities=rhos,
        young_moduli=young,
        metadata=scene_meta,
    )


def _validate_physics(index: int, row: dict, scale: np.ndarray) -> None:
    rho = float(row["rho"]["value"])
    young = float(row["E"]["value"])
    nu = float(row["nu"]["value"])
    mu = float(row["mu"]["value"])
    amp = float(row["amplitude"]["value"])
    if not np.isfinite(rho) or rho <= 0.0:
        raise SceneCompileError(f"gaussian {index} rho {rho} is not finite and positive")
    if not np.isfinite(young) or young <= 0.0:
        raise SceneCompileError(
            f"gaussian {index} Young's modulus {young} is not finite and positive"
        )
    if not np.isfinite(nu) or not (-1.0 < nu < 0.5):
        raise SceneCompileError(
            f"gaussian {index} Poisson ratio {nu} is outside (-1, 0.5); "
            "isotropic linear elasticity would not be positive definite"
        )
    if not np.isfinite(mu) or mu < 0.0:
        raise SceneCompileError(f"gaussian {index} friction mu {mu} is not finite and >= 0")
    if not np.isfinite(amp) or not (0.0 < amp <= 1.0):
        raise SceneCompileError(
            f"gaussian {index} occupancy amplitude {amp} is outside (0, 1]"
        )
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise SceneCompileError(f"gaussian {index} scale is not finite and positive")


def _normalize_quat(q: np.ndarray, index: int):
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1e-8:
        raise SceneCompileError(f"gaussian {index} quaternion has non-positive norm")
    return q / norm, abs(norm - 1.0) > 1e-6


def _rigid_transform(value):
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or np.any(~np.isfinite(matrix)):
        raise SceneCompileError("transform must be a finite 4x4 matrix")
    rot = matrix[:3, :3]
    if np.linalg.norm(rot.T @ rot - np.eye(3)) > 1e-6 or abs(float(np.linalg.det(rot)) - 1.0) > 1e-6:
        raise SceneCompileError(
            "transform must be a rigid rotation plus translation; "
            "bake scale into unit_scale instead"
        )
    # Shepperd-style via the existing matrix; build a quaternion from rot.
    quat = _matrix_to_quat(rot)
    return {"rot": rot, "t": matrix[:3, 3].copy(), "quat": quat}


def _matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _object_ids(n: int, value) -> list:
    if value is None:
        return [None] * n
    ids = np.asarray(value).reshape(-1)
    if ids.shape != (n,):
        raise SceneCompileError(f"object_id length {ids.shape[0]} != gaussian count {n}")
    out = []
    for item in ids.tolist():
        if item is None:
            out.append(None)
        else:
            out.append(int(item))
    return out


def _initial_velocities(n: int, value) -> np.ndarray:
    if value is None:
        return np.zeros((n, 3), dtype=np.float64)
    vel = np.asarray(value, dtype=np.float64)
    if vel.shape != (n, 3) or np.any(~np.isfinite(vel)):
        raise SceneCompileError("initial_velocity must be finite with shape (N, 3)")
    return vel


def _require_finite(name: str, array: np.ndarray) -> None:
    if array.size and np.any(~np.isfinite(array)):
        bad = np.argwhere(~np.isfinite(array))
        raise SceneCompileError(f"{name} has a non-finite value at index {tuple(int(v) for v in bad[0])}")


def _cloud_from_mapping(source: dict):
    conventions = source.get("conventions")
    if not isinstance(conventions, dict):
        raise SceneCompileError(
            "dict/npz input must set conventions explicitly "
            "(opacity logit|alpha, scale log|linear, quaternion wxyz|xyzw). "
            "PLY input is the path that implies the Cross3DGS conventions."
        )
    xyz = np.asarray(source["xyz"], dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise SceneCompileError("xyz must have shape (N, 3)")
    _require_finite("xyz", xyz)
    n = xyz.shape[0]
    scale = np.asarray(source["scale"], dtype=np.float64)
    opacity = np.asarray(source["opacity"], dtype=np.float64).reshape(-1)
    _require_finite("opacity", opacity)
    _require_finite("scale", scale)
    if scale.shape != (n, 3) or opacity.shape != (n,):
        raise SceneCompileError("scale or opacity does not match xyz")
    if conventions.get("scale") == "log":
        linear_scale = np.exp(scale)
    elif conventions.get("scale") == "linear":
        linear_scale = scale
    else:
        raise SceneCompileError("conventions.scale must be 'log' or 'linear'")
    if np.any(~np.isfinite(linear_scale)) or np.any(linear_scale <= 0.0):
        raise SceneCompileError("scale must be finite and positive after its convention is applied")
    if conventions.get("opacity") == "logit":
        alpha = _sigmoid(opacity)
    elif conventions.get("opacity") == "alpha":
        alpha = opacity
        if np.any((alpha <= 0.0) | (alpha > 1.0) | ~np.isfinite(alpha)):
            raise SceneCompileError("alpha opacity must be finite and in (0, 1]")
    else:
        raise SceneCompileError("conventions.opacity must be 'logit' or 'alpha'")
    if "rotation" in source and source["rotation"] is not None:
        quat = np.asarray(source["rotation"], dtype=np.float64)
        if quat.shape != (n, 4):
            raise SceneCompileError("rotation must have shape (N, 4)")
        order = conventions.get("quaternion", "wxyz")
        if order == "xyzw":
            quat = np.concatenate([quat[:, 3:4], quat[:, 0:3]], axis=1)
        elif order != "wxyz":
            raise SceneCompileError("conventions.quaternion must be 'wxyz' or 'xyzw'")
    else:
        quat = np.zeros((n, 4), dtype=np.float64)
        quat[:, 0] = 1.0
    present_dc = "f_dc" in source and source["f_dc"] is not None
    if present_dc:
        f_dc = np.asarray(source["f_dc"], dtype=np.float64)
        if f_dc.shape != (n, 3):
            raise SceneCompileError("f_dc must have shape (N, 3)")
    elif "rgb" in source and source["rgb"] is not None:
        rgb = np.asarray(source["rgb"], dtype=np.float64)
        if rgb.shape != (n, 3):
            raise SceneCompileError("rgb must have shape (N, 3)")
        f_dc = (rgb - 0.5) / SH_C0
        present_dc = True
    else:
        f_dc = np.zeros((n, 3), dtype=np.float64)
    present_rest = "f_rest" in source and source["f_rest"] is not None
    f_rest = None if not present_rest else np.asarray(source["f_rest"], dtype=np.float64)
    cloud = cloud_from_activated(xyz, linear_scale, alpha, quat, f_dc, f_rest)
    return cloud, {"f_dc": present_dc, "f_rest": present_rest}


def _cloud_from_npz(path: Path):
    with np.load(path, allow_pickle=False) as blob:
        files = set(blob.files)
        source = {key: blob[key] for key in files if key != "metadata_json"}
        meta = {}
        if "metadata_json" in files:
            raw = blob["metadata_json"]
            text = raw.item() if np.ndim(raw) == 0 else str(raw)
            meta = json.loads(str(text))
            if not isinstance(meta, dict):
                raise SceneCompileError("metadata_json must be an object")
        if "conventions" in meta and "conventions" not in source:
            source["conventions"] = meta["conventions"]
        if "object_id" in source and "object_id" not in meta:
            meta = dict(meta)
            meta["object_id"] = source["object_id"]
        if "initial_velocity" in source and "initial_velocity" not in meta:
            meta = dict(meta)
            meta["initial_velocity"] = source["initial_velocity"]
    cloud, flags = _cloud_from_mapping(source)
    return cloud, meta, flags
