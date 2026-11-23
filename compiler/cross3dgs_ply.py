"""Cross3DGS PLY exchange.

The byte layout is the writer in
``unclearness/cross3dgs`` commit ``d69f79e1431d39d7ba71e22c29065dca67a4b4ce``,
``libs/cross3dgs/src/gaussian_state.cpp`` ``SaveGaussianPly``, checked against
``tests/cpu/test_ply_export.cpp``.

- binary little-endian, one ``vertex`` element, 59 ``float`` properties
- no normals (an Inria-style ``nx, ny, nz`` file is a different layout)
- ``opacity`` is the logit of linear alpha; in-memory training state stores alpha
- ``scale_*`` is the log of linear principal scale
- ``rot_0..3`` is ``(w, x, y, z)`` (``rot_0`` is the real part)
- ``f_rest_0..44`` is channel-major: index ``channel * 15 + coefficient``
- SH DC uses ``kShC0 = 0.28209479177387814``

This module reads and writes that exchange file. It is not a renderer and it
is not called from the integrator.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from compiler.errors import SceneCompileError

# SaveGaussianPly clamps linear alpha and scale with this floor before log/logit.
ACTIVATION_FLOOR = 1.0e-6
SH_C0 = 0.28209479177387814
SH_REST_COEFFS = 15
SH_REST_CHANNELS = 3

CROSS3DGS_PLY_PROPERTIES: tuple[str, ...] = (
    ("x", "y", "z")
    + tuple(f"f_dc_{i}" for i in range(3))
    + tuple(f"f_rest_{i}" for i in range(SH_REST_COEFFS * SH_REST_CHANNELS))
    + ("opacity",)
    + tuple(f"scale_{i}" for i in range(3))
    + tuple(f"rot_{i}" for i in range(4))
)

_REQUIRED = ("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3")


@dataclass
class Cross3DGSCloud:
    """File-domain arrays. Shapes are ``(N, ...)`` and dtype is float32.

    These are the PLY numbers, not activated physics quantities. Writing the
    cloud back reproduces the exchange file; it does not re-encode the
    diagonal occupancy kernel used by the integrator.
    """

    xyz: np.ndarray
    f_dc: np.ndarray
    f_rest: np.ndarray
    opacity_logit: np.ndarray
    log_scale: np.ndarray
    rot_wxyz: np.ndarray

    def __post_init__(self):
        n = int(self.xyz.shape[0])
        for name, array, tail in (
            ("xyz", self.xyz, (3,)),
            ("f_dc", self.f_dc, (3,)),
            ("f_rest", self.f_rest, (45,)),
            ("opacity_logit", self.opacity_logit, ()),
            ("log_scale", self.log_scale, (3,)),
            ("rot_wxyz", self.rot_wxyz, (4,)),
        ):
            if array.dtype != np.float32:
                raise SceneCompileError(f"{name} must be float32, got {array.dtype}")
            expect = (n,) + tail
            if array.shape != expect:
                raise SceneCompileError(
                    f"{name} shape {array.shape} does not match xyz count {n}"
                )

    @property
    def count(self) -> int:
        return int(self.xyz.shape[0])


def cloud_from_activated(xyz, scale, alpha, quat_wxyz, f_dc, f_rest=None) -> Cross3DGSCloud:
    """Build a file-domain cloud the way ``SaveGaussianPly`` encodes state.

    ``scale`` and ``alpha`` are linear. ``quat_wxyz`` is ``(w, x, y, z)``.
    Values are cast to float32 after the same floor used by Cross3DGS.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    f_dc = np.asarray(f_dc, dtype=np.float64)
    n = xyz.shape[0]
    if scale.shape != (n, 3) or quat.shape != (n, 4) or f_dc.shape != (n, 3):
        raise SceneCompileError("activated cloud arrays do not share one vertex count")
    if np.any(~np.isfinite(xyz)) or np.any(~np.isfinite(scale)) or np.any(~np.isfinite(alpha)):
        raise SceneCompileError("activated cloud has a non-finite value")
    alpha_c = np.clip(alpha, ACTIVATION_FLOOR, 1.0 - ACTIVATION_FLOOR)
    logit = np.log(alpha_c / (1.0 - alpha_c))
    log_scale = np.log(np.maximum(scale, ACTIVATION_FLOOR))
    if f_rest is None:
        rest = np.zeros((n, 45), dtype=np.float32)
    else:
        rest = np.asarray(f_rest, dtype=np.float32)
        if rest.shape != (n, 45):
            raise SceneCompileError(f"f_rest shape {rest.shape}, expected {(n, 45)}")
    return Cross3DGSCloud(
        xyz=xyz.astype(np.float32),
        f_dc=f_dc.astype(np.float32),
        f_rest=rest,
        opacity_logit=logit.astype(np.float32),
        log_scale=log_scale.astype(np.float32),
        rot_wxyz=quat.astype(np.float32),
    )


def write_cross3dgs_ply(path, cloud: Cross3DGSCloud) -> None:
    """Write ``SaveGaussianPly``'s header and raw file-domain floats."""
    path = Path(path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    n = cloud.count
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header.extend(f"property float {name}" for name in CROSS3DGS_PLY_PROPERTIES)
    header.append("end_header")
    payload = np.concatenate(
        [
            cloud.xyz,
            cloud.f_dc,
            cloud.f_rest,
            cloud.opacity_logit.reshape(n, 1),
            cloud.log_scale,
            cloud.rot_wxyz,
        ],
        axis=1,
    ).astype("<f4", copy=False)
    if payload.shape != (n, len(CROSS3DGS_PLY_PROPERTIES)):
        raise SceneCompileError("internal PLY row width does not match the Cross3DGS header")
    with path.open("wb") as handle:
        handle.write(("\n".join(header) + "\n").encode("ascii"))
        if n:
            handle.write(payload.tobytes())


def read_cross3dgs_ply(path) -> Cross3DGSCloud:
    """Read a Cross3DGS exchange PLY by property name.

    Extra named properties (for example Inria normals) are ignored. Missing
    ``f_dc`` or ``f_rest_*`` become zeros and are reported by the compiler as
    absent appearance, not as a guess that the file is a different schema.
    A non-little-endian format is an error.
    """
    path = Path(path)
    with path.open("rb") as handle:
        magic = handle.readline()
        if magic.strip() != b"ply":
            raise SceneCompileError(f"{path} is not a PLY file")
        fmt = None
        count = None
        props: list[tuple[str, str]] = []
        while True:
            raw = handle.readline()
            if not raw:
                raise SceneCompileError(f"{path} header ended before end_header")
            line = raw.decode("ascii", errors="strict").strip()
            if line.startswith("comment") or line.startswith("obj_info"):
                continue
            if line.startswith("format "):
                fmt = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                if len(parts) != 3:
                    raise SceneCompileError(f"bad PLY element line: {line}")
                if parts[1] == "vertex":
                    count = int(parts[2])
                else:
                    count = count  # a later element is not part of the vertex block
            elif line.startswith("property "):
                parts = line.split()
                if parts[1] == "list":
                    raise SceneCompileError(
                        "Cross3DGS vertex properties are scalar floats; "
                        f"got a list property in {path}"
                    )
                if len(parts) != 3:
                    raise SceneCompileError(f"bad PLY property line: {line}")
                props.append((parts[1], parts[2]))
            elif line == "end_header":
                break
            else:
                raise SceneCompileError(f"unsupported PLY header line: {line}")
        if fmt != "binary_little_endian":
            raise SceneCompileError(
                "Cross3DGS SaveGaussianPly writes format binary_little_endian; "
                f"{path} says {fmt!r}"
            )
        if count is None:
            raise SceneCompileError(f"{path} has no vertex element")
        names = [name for _, name in props]
        missing = [name for name in _REQUIRED if name not in names]
        if missing:
            raise SceneCompileError(
                f"{path} is missing Cross3DGS properties {missing}. "
                "Required file-domain fields are x,y,z, opacity logit, "
                "log scale_0..2, and rot_0..3 as wxyz."
            )
        if any(typ != "float" for typ, _ in props):
            raise SceneCompileError(
                f"{path} uses a non-float vertex property; "
                "Cross3DGS exchange properties are float32"
            )
        dtype = np.dtype([(name, "<f4") for name in names])
        if count:
            raw = np.fromfile(handle, dtype=dtype, count=count)
            if raw.shape[0] != count:
                raise SceneCompileError(
                    f"{path} ended after {raw.shape[0]} vertices, header says {count}"
                )
        else:
            raw = np.zeros(0, dtype=dtype)

    def take(name: str) -> np.ndarray:
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        return np.asarray(raw[name], dtype=np.float32)

    def take_group(prefix: str, width: int) -> np.ndarray:
        data = np.zeros((count, width), dtype=np.float32)
        for i in range(width):
            key = f"{prefix}{i}"
            if key in names and count:
                data[:, i] = raw[key]
        return data

    if count:
        xyz = np.stack([take("x"), take("y"), take("z")], axis=1)
        log_scale = np.stack([take("scale_0"), take("scale_1"), take("scale_2")], axis=1)
        rot = np.stack([take(f"rot_{i}") for i in range(4)], axis=1)
    else:
        xyz = np.zeros((0, 3), dtype=np.float32)
        log_scale = np.zeros((0, 3), dtype=np.float32)
        rot = np.zeros((0, 4), dtype=np.float32)
    cloud = Cross3DGSCloud(
        xyz=xyz.astype(np.float32, copy=False),
        f_dc=take_group("f_dc_", 3),
        f_rest=take_group("f_rest_", 45),
        opacity_logit=take("opacity"),
        log_scale=log_scale.astype(np.float32, copy=False),
        rot_wxyz=rot.astype(np.float32, copy=False),
    )
    # Missing appearance channels stay zero and must not drive the color heuristic.
    cloud.file_has_f_dc = all(f"f_dc_{i}" in names for i in range(3))
    cloud.file_has_f_rest = all(f"f_rest_{i}" in names for i in range(45))
    return cloud
