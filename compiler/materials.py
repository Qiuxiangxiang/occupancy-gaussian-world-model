"""Appearance-to-physics guesses.

Color, opacity, and splat scale do not identify mass density or Young's
modulus. The table below is a bounded initial guess. A per-object override
replaces a palette guess, and a per-gaussian override replaces both. Nothing
here is a calibration, and nothing is inferred by optimizing against motion.
"""
from __future__ import annotations

import numpy as np

import config as C
from compiler.cross3dgs_ply import SH_C0
from compiler.errors import SceneCompileError

DISCLAIMER = (
    "Color, opacity, and splat scale do not identify mass density or Young's "
    "modulus. Mapped numbers are bounded initial guesses. Overrides replace "
    "guesses; they are not a fit to observed motion."
)


def default_mapping_table() -> dict:
    """Copyable guess table. Palette moduli sit inside the heuristic clamp."""
    return {
        "amplitude": {"bias": 0.25, "gain": 0.75, "lo": 0.05, "hi": 1.0},
        # Applied only when the source is the color heuristic.
        "heuristic_bounds": {"rho": (200.0, 8000.0), "E": (1.0e4, 1.0e9)},
        "palette": {
            "ceramic": {"rgb": (0.90, 0.90, 0.86), "rho": 2500.0, "E": 1.0e9, "nu": 0.22, "mu": 0.25},
            "metal": {"rgb": (0.72, 0.74, 0.78), "rho": 4000.0, "E": 1.0e9, "nu": 0.28, "mu": 0.20},
            "plastic": {"rgb": (0.20, 0.35, 0.85), "rho": 1200.0, "E": 1.0e6, "nu": 0.35, "mu": 0.30},
            "rubber": {"rgb": (0.80, 0.15, 0.15), "rho": 1100.0, "E": 2.0e5, "nu": 0.45, "mu": 0.70},
            "wood": {"rgb": (0.55, 0.35, 0.15), "rho": 700.0, "E": 5.0e8, "nu": 0.30, "mu": 0.40},
        },
        "default": {
            "rho": float(C.RHO),
            "E": float(C.E_YOUNG),
            "nu": float(C.NU),
            "mu": float(C.FRICTION_MU),
        },
    }


def sh_dc_to_rgb(f_dc: np.ndarray) -> np.ndarray:
    """View-independent color used by Cross3DGS: ``0.5 + SH_C0 * f_dc``.

    Values outside ``[0, 1]`` are clipped only for the palette lookup.
    """
    rgb = 0.5 + SH_C0 * np.asarray(f_dc, dtype=np.float64)
    return np.clip(rgb, 0.0, 1.0)


def _lookup(table: dict, key):
    if key is None:
        return {}
    text = str(key)
    if text in table:
        return table[text]
    if key in table:
        return table[key]
    return {}


def _nearest_material(rgb: np.ndarray, palette: dict) -> str:
    best_name = None
    best_d = None
    for name in sorted(palette):
        ref = np.asarray(palette[name]["rgb"], dtype=np.float64)
        dist = float(np.sum((rgb - ref) ** 2))
        if best_name is None or dist < best_d - 1e-15 or (
            abs(dist - best_d) <= 1e-15 and name < best_name
        ):
            best_name = name
            best_d = dist
    if best_name is None:
        raise SceneCompileError("color heuristic palette is empty")
    return best_name


def _prop(value, source: str, material: str | None, clamped: bool) -> dict:
    return {
        "value": value,
        "source": source,
        "material": material,
        "clamped": bool(clamped),
    }


def resolve_materials(n: int, object_ids: list, rgb: np.ndarray | None,
                      metadata: dict, table: dict | None = None) -> list[dict]:
    """Resolve rho, E, nu, mu, amplitude, and boundary for each gaussian.

    Precedence, highest first: per-gaussian override, per-object override,
    named material on the gaussian, named material on the object, nearest
    palette color, then ``config.py`` defaults. Heuristic clamps apply only
    to color-chosen rho and E.
    """
    table = table or default_mapping_table()
    meta = metadata or {}
    g_over = meta.get("gaussian_overrides") or {}
    o_over = meta.get("object_overrides") or {}
    g_bounds = meta.get("gaussian_boundaries") or {}
    o_bounds = meta.get("object_boundaries") or {}
    amp_map = dict(table["amplitude"])
    amp_map.update(meta.get("amplitude_map") or {})
    out = []
    for index in range(n):
        oid = object_ids[index]
        grow = _lookup(g_over, index)
        orow = _lookup(o_over, oid) if oid is not None else {}
        if "material" in grow:
            material = str(grow["material"])
            name_source = "gaussian_material"
        elif "material" in orow:
            material = str(orow["material"])
            name_source = "object_material"
        elif rgb is not None:
            material = _nearest_material(rgb[index], table["palette"])
            name_source = "color_heuristic"
        else:
            material = None
            name_source = "default"
        if material is None:
            base = dict(table["default"])
        else:
            if material not in table["palette"]:
                raise SceneCompileError(
                    f"gaussian {index} names unknown material {material!r}"
                )
            base = {k: table["palette"][material][k] for k in ("rho", "E", "nu", "mu")}
        resolved = {"material": material, "name_source": name_source}
        for prop in ("rho", "E", "nu", "mu"):
            if prop in grow:
                resolved[prop] = _prop(float(grow[prop]), "gaussian_override", material, False)
            elif prop in orow:
                resolved[prop] = _prop(float(orow[prop]), "object_override", material, False)
            else:
                value = float(base[prop])
                clamped = False
                if name_source == "color_heuristic" and prop in table["heuristic_bounds"]:
                    lo, hi = table["heuristic_bounds"][prop]
                    clamped = value < lo or value > hi
                    value = float(min(max(value, lo), hi))
                source = name_source if name_source != "default" else "default"
                resolved[prop] = _prop(value, source, material, clamped)
        alpha = None  # filled by the caller via amplitude override path
        if "occupancy_amplitude" in grow:
            amp = float(grow["occupancy_amplitude"])
            resolved["amplitude"] = _prop(amp, "gaussian_override", material, False)
        elif "occupancy_amplitude" in orow:
            amp = float(orow["occupancy_amplitude"])
            resolved["amplitude"] = _prop(amp, "object_override", material, False)
        else:
            resolved["amplitude"] = None
            resolved["amplitude_needs_alpha"] = amp_map
        gb = _lookup(g_bounds, index)
        ob = _lookup(o_bounds, oid) if oid is not None else {}
        if gb:
            resolved["boundary"] = _boundary(gb, "gaussian_override")
        elif ob:
            resolved["boundary"] = _boundary(ob, "object_override")
        else:
            resolved["boundary"] = _boundary({"kind": "free"}, "default")
        resolved["alpha_slot"] = alpha
        out.append(resolved)
    return out


def amplitude_from_alpha(alpha: float, amp_map: dict) -> dict:
    raw = float(amp_map["bias"] + amp_map["gain"] * alpha)
    lo = float(amp_map["lo"])
    hi = float(amp_map["hi"])
    value = float(min(max(raw, lo), hi))
    return _prop(value, "alpha_heuristic", None, value != raw)


def _boundary(spec: dict, source: str) -> dict:
    kind = str(spec.get("kind", "free"))
    if kind not in ("free", "fixed", "prescribed_velocity", "embedded"):
        raise SceneCompileError(
            f"boundary kind {kind!r} is not free, fixed, prescribed_velocity, or embedded"
        )
    velocity = spec.get("velocity")
    if kind == "prescribed_velocity":
        if velocity is None:
            raise SceneCompileError("prescribed_velocity boundary needs a velocity")
        velocity = [float(v) for v in velocity]
        if len(velocity) != 3 or not all(np.isfinite(velocity)):
            raise SceneCompileError("prescribed velocity must be three finite numbers")
    else:
        velocity = None
    return {
        "kind": kind,
        "source": source,
        "velocity": velocity,
        "support_stiffness": spec.get("support_stiffness"),
        "dof_mask": spec.get("dof_mask"),
    }
