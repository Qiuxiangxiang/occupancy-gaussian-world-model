"""Embedded volume supports for gaussians planted in a support.

The repository does not name which bodies are rooted. The rule used here is
frozen at compile time and is not updated by contact events:

1. Boundary kind ``embedded`` on that gaussian or its object.
2. ``embedded_indices`` / ``embedded_objects`` in the sidecar.
3. If ``ground_plane`` is set, every gaussian whose compiled center lies on
   the support side of that plane (``normal · x <= offset``).

A later crossing of the plane does not add or remove a support. The set is
the same at step 0 and on a step with no collision.

The support is a bilateral quadratic energy on the translational DOFs Phase 0
actually integrates. There is no rotational DOF on ``OccupancyGaussian``.

    E = 1/2 k || M (x - x_rest) ||^2     joules, if x is in meters and k in N/m
    F = -k M (x - x_rest)                newtons

``M`` is a diagonal 0/1 mask, default all three axes. ``k`` is
``support_stiffness`` when given, otherwise ``E_young * mean(scale)``
(Pa · m = N/m). That product is a modeling default, not an identification.

This is not a surface-contact penalty. Contact forces exist only while the
1-sigma predicate is true and are applied through the contact graph. The
support force is added on every force evaluation, including the Verlet second
kick, whether or not any contact pair exists. ``apply_boundary_conditions``
does not move these bodies; a fixed boundary is a separate Dirichlet reset.

The stiffness is also added on the diagonal of a copy of ``K`` after
``assemble_K_M``. That function itself is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from compiler.errors import SceneCompileError


@dataclass
class EmbeddedSupport:
    index: int
    rest_m: np.ndarray
    stiffness_Npm: float
    mask: np.ndarray
    reasons: tuple


@dataclass
class EmbeddedSupportSet:
    supports: list[EmbeddedSupport] = field(default_factory=list)
    evaluations: int = 0

    def indices(self) -> list[int]:
        return [support.index for support in self.supports]

    def force(self, positions) -> np.ndarray:
        """Support force at these positions. Always the embedded law.

        Calling this counts as the constraint being in the residual, even
        when the force is zero because the body is at its rest pose.
        """
        self.evaluations += 1
        pos = np.asarray(positions, dtype=np.float64)
        out = np.zeros_like(pos, dtype=np.float64)
        for support in self.supports:
            delta = (pos[support.index] - support.rest_m) * support.mask
            out[support.index] -= support.stiffness_Npm * delta
        return out

    def energy(self, positions) -> float:
        pos = np.asarray(positions, dtype=np.float64)
        total = 0.0
        for support in self.supports:
            delta = (pos[support.index] - support.rest_m) * support.mask
            total += 0.5 * support.stiffness_Npm * float(np.dot(delta, delta))
        return float(total)

    def stiffness_matrix(self, n_gaussians: int) -> np.ndarray:
        """Diagonal support contribution, shape ``(3N, 3N)``, units N/m."""
        matrix = np.zeros((3 * n_gaussians, 3 * n_gaussians), dtype=np.float64)
        for support in self.supports:
            for axis in range(3):
                if support.mask[axis] == 0.0:
                    continue
                dof = 3 * support.index + axis
                matrix[dof, dof] += support.stiffness_Npm
        return matrix

    def add_to_stiffness(self, stiffness: np.ndarray) -> np.ndarray:
        """Return ``K + K_support`` without modifying the input array."""
        added = np.array(stiffness, dtype=np.float64, copy=True)
        added += self.stiffness_matrix(added.shape[0] // 3)
        return added


def _mask(value, index: int) -> np.ndarray:
    if value is None:
        return np.ones(3, dtype=np.float64)
    mask = np.asarray(value, dtype=np.float64).reshape(-1)
    if mask.shape != (3,) or np.any(~np.isfinite(mask)):
        raise SceneCompileError(f"gaussian {index} dof_mask must be three finite numbers")
    if np.any((mask != 0.0) & (mask != 1.0)):
        raise SceneCompileError(f"gaussian {index} dof_mask entries must be 0 or 1")
    if float(np.sum(mask)) <= 0.0:
        raise SceneCompileError(f"gaussian {index} embedded support masks every DOF off")
    return mask


def _stiffness(explicit, gaussian, index: int) -> float:
    if explicit is None:
        value = float(gaussian.E) * float(np.mean(gaussian.scale))
    else:
        value = float(explicit)
    if not np.isfinite(value) or value <= 0.0:
        raise SceneCompileError(
            f"gaussian {index} support stiffness {value} must be finite and positive (N/m)"
        )
    return value


def build_embedded_supports(gaussians, boundaries, object_ids, metadata) -> EmbeddedSupportSet:
    """Freeze the rooted set from the compiled pose and the sidecar."""
    meta = metadata or {}
    reasons: dict[int, list[str]] = {}
    stiffness_override: dict[int, float] = {}
    masks: dict[int, np.ndarray] = {}

    def add(index: int, reason: str) -> None:
        if index < 0 or index >= len(gaussians):
            raise SceneCompileError(f"embedded index {index} is outside 0..{len(gaussians) - 1}")
        reasons.setdefault(index, [])
        if reason not in reasons[index]:
            reasons[index].append(reason)

    for bc in boundaries:
        if bc.kind == "embedded":
            add(bc.index, bc.source if bc.source.startswith("embedded") else f"boundary:{bc.source}")
            if bc.support_stiffness is not None:
                stiffness_override[bc.index] = float(bc.support_stiffness)
            if bc.dof_mask is not None:
                masks[bc.index] = _mask(bc.dof_mask, bc.index)
    for raw in meta.get("embedded_indices") or []:
        add(int(raw), "embedded_indices")
    wanted_objects = {int(item) for item in (meta.get("embedded_objects") or [])}
    for index, object_id in enumerate(object_ids):
        if object_id is not None and int(object_id) in wanted_objects:
            add(index, "embedded_objects")
    plane = meta.get("ground_plane")
    if plane is not None:
        normal = np.asarray(plane.get("normal", [0.0, 0.0, 1.0]), dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all() or float(np.linalg.norm(normal)) <= 0.0:
            raise SceneCompileError("ground_plane.normal must be a finite nonzero 3-vector")
        normal = normal / float(np.linalg.norm(normal))
        offset = float(plane.get("offset", 0.0))
        if not np.isfinite(offset):
            raise SceneCompileError("ground_plane.offset must be finite")
        for index, gaussian in enumerate(gaussians):
            if float(normal @ gaussian.pos) <= offset:
                add(index, "ground_plane")
    per_index = meta.get("embedded_stiffness") or {}
    global_k = meta.get("support_stiffness")
    global_mask = meta.get("dof_mask")
    supports = []
    for index in sorted(reasons):
        explicit = stiffness_override.get(index)
        if explicit is None and str(index) in per_index:
            explicit = per_index[str(index)]
        elif explicit is None and index in per_index:
            explicit = per_index[index]
        elif explicit is None and global_k is not None:
            explicit = global_k
        mask = masks.get(index)
        if mask is None and global_mask is not None:
            mask = _mask(global_mask, index)
        elif mask is None:
            mask = _mask(None, index)
        supports.append(EmbeddedSupport(
            index=index,
            rest_m=np.array(gaussians[index].pos, dtype=np.float64, copy=True),
            stiffness_Npm=_stiffness(explicit, gaussians[index], index),
            mask=mask,
            reasons=tuple(reasons[index]),
        ))
    return EmbeddedSupportSet(supports=supports)
