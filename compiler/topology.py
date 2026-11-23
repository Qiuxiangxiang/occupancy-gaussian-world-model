"""Symbolic active-set labels on the chronological event log.

These are constraint-switch records, not samples of every timestep. The
integrator is unchanged: penalty forces still use the instantaneous 1-sigma
test in ``contact_between``. The labels below debounce that test so chatter
inside a band is not a new topology event.

The phrase "incompressibility constraint switching" is an active-set switch
of a constraint this model actually has: a penalty contact, the M2 embedded
quadratic support, or the existing Coulomb term. ``nu`` only enters the Lamé
parameters inside ``pairwise_stiffness``. Nothing here toggles a volumetric
incompressibility constraint, and the penalty is not a complementarity
solver. Every event says ``volumetric_incompressibility: false``.

Normal axis (dimensionless penetration / min mean scale, the same ratio as
``eval_penetration``):

- enter a logged contact when the ratio >= 0.02
- leave only when the ratio <= 0
- ``point``: one logged penalty pair, ratio below the area band
- ``area``: ratio >= 0.20, or at least two logged pairs between the same
  object pair; leave the depth rule only when the ratio < 0.12
- ``self_penetration``: both gaussians share a non-null object id

A missing object id does not invent a body, so two unsegmented gaussians are
not self-penetration.

Tangential axis (m/s), using the relative tangent already passed to
``penalty_force``. No second friction law.

- ``separated`` while the normal label is separated
- ``free`` when mu <= 0: sticking is undefined because the Coulomb term is
  identically zero. This is not called sliding and not called sticking
- ``sticking`` when mu > 0 and speed <= 1e-3, leaving only at speed >= 2e-3
- ``sliding`` when mu > 0 and speed stays at or above that upper edge

Modal axis: on a normal, tangential, or embedded-support switch, if tracked
modes are supplied, score_i = g_i^2 / sum_j g_j^2 with g_i = φ_i · f and f
the penalty (or support) force. A mode is excited when its frequency is at
least 1 Hz and its score is at least 0.20. Identity is the column order from
M2's tracker. Participation is not a new force.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compiler.events import EventLog
from occupancy_gaussian_framework import penalty_force

# Dimensionless penetration ratio, matching eval_penetration's denominator.
PEN_RATIO_ENTER = 0.02
PEN_RATIO_RELEASE = 0.0
AREA_PEN_RATIO_ENTER = 0.20
AREA_PEN_RATIO_RELEASE = 0.12
# Tangential speed, meters per second.
STICK_SPEED_ENTER_M_S = 1.0e-3
STICK_SPEED_LEAVE_M_S = 2.0e-3
# Fraction of the sum of squared modal forces. Frequencies below this are
# rigid-body zeros and are not called participation.
PARTICIPATION_MIN = 0.20
PARTICIPATION_FREQ_MIN_HZ = 1.0

SCHEMA = "m3-constraint-switch-v1"


def thresholds() -> dict:
    return {
        "pen_ratio_enter": PEN_RATIO_ENTER,
        "pen_ratio_release": PEN_RATIO_RELEASE,
        "area_pen_ratio_enter": AREA_PEN_RATIO_ENTER,
        "area_pen_ratio_release": AREA_PEN_RATIO_RELEASE,
        "stick_speed_enter_m_s": STICK_SPEED_ENTER_M_S,
        "stick_speed_leave_m_s": STICK_SPEED_LEAVE_M_S,
        "participation_min": PARTICIPATION_MIN,
        "participation_freq_min_hz": PARTICIPATION_FREQ_MIN_HZ,
        "pen_ratio_unit": "penetration / min(mean scale), dimensionless",
        "speed_unit": "m/s",
        "participation_unit": "fraction of sum of squared modal forces",
    }


@dataclass
class _PairState:
    engaged: bool = False
    normal: str = "separated"
    tangential: str = "separated"
    excited: tuple = ()


def pair_geometry(gaussians, i: int, j: int, velocities=None) -> dict:
    """1-sigma geometry shared with ``contact_between``, without its gate."""
    if j < i:
        i, j = j, i
    g1, g2 = gaussians[i], gaussians[j]
    delta = g2.pos - g1.pos
    dist = float(np.linalg.norm(delta) + 1e-12)
    r1 = float(np.sqrt(delta @ g1.cov @ delta) / dist)
    r2 = float(np.sqrt(delta @ g2.cov @ delta) / dist)
    pen = (r1 + r2) - dist
    normal = delta / dist
    if velocities is None:
        rel = np.zeros(3)
    else:
        rel = np.asarray(velocities[j] - velocities[i], dtype=np.float64)
    tangent = rel - float(np.dot(rel, normal)) * normal
    scale = min(float(np.mean(g1.scale)), float(np.mean(g2.scale)))
    return {
        "i": i,
        "j": j,
        "pair_id": f"{i}-{j}",
        "pen_ratio": float(pen / max(scale, 1e-12)),
        "speed_m_s": float(np.linalg.norm(tangent)),
        "normal": normal,
        "penetration": float(pen),
        "rel_vel": rel,
    }


def modal_participation(force, vectors, frequencies_hz) -> dict:
    """Squared modal-force fractions for tracked columns of ``vectors``."""
    force = np.asarray(force, dtype=np.float64).reshape(-1)
    vectors = np.asarray(vectors, dtype=np.float64)
    freqs = [float(value) for value in frequencies_hz]
    raw = []
    for index in range(vectors.shape[1]):
        modal_force = float(vectors[:, index] @ force)
        raw.append(modal_force * modal_force)
    total = float(sum(raw))
    scores = []
    excited = []
    for index, value in enumerate(raw):
        score = 0.0 if total <= 0.0 else value / total
        freq = freqs[index] if index < len(freqs) else 0.0
        scores.append({"tracked_id": index, "frequency_hz": freq, "score": score})
        if freq >= PARTICIPATION_FREQ_MIN_HZ and score >= PARTICIPATION_MIN:
            excited.append(index)
    return {"scores": scores, "excited_tracked_ids": excited}


class SymbolicTopology:
    """Debounced labels. ``observe`` appends a switch only when a label changes."""

    def __init__(self, log: EventLog, object_ids, mu: float = 0.0,
                 embedded_indices=None):
        self.log = log
        self.object_ids = list(object_ids)
        self.mu = float(mu)
        self.embedded_indices = [int(index) for index in (embedded_indices or [])]
        self.pairs: dict[tuple[int, int], _PairState] = {}
        self._support_logged = False
        self.full_rebuilds = 0

    def observe(self, time_s: float, gaussians, contacts, velocities=None,
                modes=None, support_force=None) -> list[dict]:
        """Classify the contact set the caller already computed.

        ``contacts`` is the instantaneous penalty set. This method does not
        rebuild it. ``modes`` is optional ``{vectors, frequencies_hz}`` in
        tracked-column order.
        """
        events = []
        if not self._support_logged:
            self._support_logged = True
            if self.embedded_indices:
                events.extend(self._emit_support(time_s, gaussians, support_force, modes))
        raw = {}
        for pair in contacts:
            key = (min(int(pair.i), int(pair.j)), max(int(pair.i), int(pair.j)))
            raw[key] = pair
        keys = set(raw)
        for key, state in self.pairs.items():
            if state.engaged or state.normal != "separated":
                keys.add(key)
        measured = {key: pair_geometry(gaussians, key[0], key[1], velocities) for key in keys}
        for key in sorted(keys):
            state = self.pairs.setdefault(key, _PairState())
            state.engaged = _engaged(state.engaged, measured[key]["pen_ratio"])
        counts: dict[tuple, int] = {}
        for key, state in self.pairs.items():
            if state.engaged:
                bucket = _object_pair_key(self.object_ids, key)
                counts[bucket] = counts.get(bucket, 0) + 1
        for key in sorted(self.pairs):
            state = self.pairs[key]
            geo = measured.get(key)
            if geo is None:
                continue
            multi = counts.get(_object_pair_key(self.object_ids, key), 0)
            self_pen = _is_self(self.object_ids, key)
            normal = _normal_label(state.engaged, state.normal, geo["pen_ratio"], self_pen, multi)
            tangential = _tangential_label(state.tangential, normal != "separated", self.mu, geo["speed_m_s"])
            changed = []
            if normal != state.normal:
                changed.append(self._emit(
                    time_s, "normal", state.normal, normal, geo, "penalty_contact",
                ))
            if tangential != state.tangential:
                physics = "coulomb_penalty" if self.mu > 0.0 else "no_tangential_constraint"
                changed.append(self._emit(
                    time_s, "tangential", state.tangential, tangential, geo, physics,
                ))
            state.normal = normal
            state.tangential = tangential
            events.extend(changed)
            if changed and modes is not None:
                excited_event = self._emit_participation(
                    time_s, geo, gaussians, modes, state.excited,
                )
                if excited_event is not None:
                    state.excited = tuple(excited_event["after"])
                    events.append(excited_event)
        return events

    def _emit(self, time_s, axis, before, after, geo, physics) -> dict:
        return self.log.append(
            time_s, "constraint_switch",
            schema=SCHEMA,
            axis=axis,
            physics=physics,
            volumetric_incompressibility=False,
            active_set_meaning=(
                "debounced symbolic label; the penalty force still uses the "
                "instantaneous 1-sigma predicate"
            ),
            pair_id=geo["pair_id"],
            gaussian_ids=[geo["i"], geo["j"]],
            object_ids=[_oid(self.object_ids, geo["i"]), _oid(self.object_ids, geo["j"])],
            before=before,
            after=after,
            pen_ratio=geo["pen_ratio"],
            tangential_speed_m_s=geo["speed_m_s"],
            mu=self.mu,
            embedded_support_indices=list(self.embedded_indices),
            thresholds=thresholds(),
        )

    def _emit_support(self, time_s, gaussians, support_force, modes) -> list[dict]:
        events = []
        for index in self.embedded_indices:
            events.append(self.log.append(
                time_s, "constraint_switch",
                schema=SCHEMA,
                axis="embedded_support",
                physics="embedded_quadratic",
                volumetric_incompressibility=False,
                active_set_meaning="frozen M2 support, announced once at the first observe",
                pair_id=f"support:{index}",
                gaussian_ids=[index],
                object_ids=[_oid(self.object_ids, index)],
                before="inactive",
                after="active",
                pen_ratio=None,
                tangential_speed_m_s=None,
                mu=self.mu,
                embedded_support_indices=list(self.embedded_indices),
                thresholds=thresholds(),
            ))
        if modes is not None and support_force is not None:
            scored = modal_participation(
                support_force, modes["vectors"], modes["frequencies_hz"],
            )
            events.append(self.log.append(
                time_s, "constraint_switch",
                schema=SCHEMA,
                axis="modal_participation",
                physics="modal_force_fraction",
                volumetric_incompressibility=False,
                pair_id="support",
                gaussian_ids=list(self.embedded_indices),
                object_ids=[_oid(self.object_ids, index) for index in self.embedded_indices],
                before=[],
                after=scored["excited_tracked_ids"],
                scores=scored["scores"],
                embedded_support_indices=list(self.embedded_indices),
                thresholds=thresholds(),
            ))
        return events

    def _emit_participation(self, time_s, geo, gaussians, modes, previous) -> dict | None:
        from occupancy_gaussian_framework import ContactPair
        pair = ContactPair(geo["i"], geo["j"], geo["normal"], max(geo["penetration"], 0.0), geo["rel_vel"])
        left, right = penalty_force(pair, gaussians, mu=self.mu)
        force = np.zeros(3 * len(gaussians))
        force[3 * geo["i"]:3 * geo["i"] + 3] = left
        force[3 * geo["j"]:3 * geo["j"] + 3] = right
        scored = modal_participation(force, modes["vectors"], modes["frequencies_hz"])
        after = tuple(scored["excited_tracked_ids"])
        if after == tuple(previous):
            return None
        return self.log.append(
            time_s, "constraint_switch",
            schema=SCHEMA,
            axis="modal_participation",
            physics="modal_force_fraction",
            volumetric_incompressibility=False,
            pair_id=geo["pair_id"],
            gaussian_ids=[geo["i"], geo["j"]],
            object_ids=[_oid(self.object_ids, geo["i"]), _oid(self.object_ids, geo["j"])],
            before=list(previous),
            after=list(after),
            scores=scored["scores"],
            mu=self.mu,
            embedded_support_indices=list(self.embedded_indices),
            thresholds=thresholds(),
        )


def _oid(object_ids, index):
    if index >= len(object_ids):
        return None
    value = object_ids[index]
    return None if value is None else int(value)


def _object_pair_key(object_ids, key: tuple[int, int]) -> tuple:
    def one(index):
        oid = _oid(object_ids, index)
        return ("obj", oid) if oid is not None else ("gaussian", index)
    return tuple(sorted((one(key[0]), one(key[1]))))


def _is_self(object_ids, key: tuple[int, int]) -> bool:
    left = _oid(object_ids, key[0])
    right = _oid(object_ids, key[1])
    return left is not None and left == right


def _engaged(previous: bool, pen_ratio: float) -> bool:
    if not previous:
        return pen_ratio >= PEN_RATIO_ENTER
    return pen_ratio > PEN_RATIO_RELEASE


def _normal_label(engaged: bool, previous: str, pen_ratio: float,
                  self_penetration: bool, multi: int) -> str:
    if not engaged:
        return "separated"
    if self_penetration:
        return "self_penetration"
    deep = pen_ratio >= AREA_PEN_RATIO_ENTER
    if previous == "area":
        deep = pen_ratio >= AREA_PEN_RATIO_RELEASE
    if deep or multi >= 2:
        return "area"
    return "point"


def _tangential_label(previous: str, engaged: bool, mu: float, speed: float) -> str:
    if not engaged:
        return "separated"
    if mu <= 0.0:
        return "free"
    if previous in ("separated", "free"):
        return "sticking" if speed <= STICK_SPEED_ENTER_M_S else "sliding"
    if previous == "sticking":
        return "sliding" if speed >= STICK_SPEED_LEAVE_M_S else "sticking"
    if previous == "sliding":
        return "sticking" if speed <= STICK_SPEED_ENTER_M_S else "sliding"
    return "sliding" if speed >= STICK_SPEED_LEAVE_M_S else "sticking"


def example_sequences() -> dict:
    """Canonical switches used by the M3 artifact. No RNG."""
    from occupancy_gaussian_framework import OccupancyGaussian, contact_between, modal_analysis

    def body(pos, scale=0.2):
        return OccupancyGaussian(pos=np.array(pos, dtype=np.float64), scale=np.ones(3) * scale, mass=1.0)

    def at(ratio, scale=0.2):
        return [body([0.0, 0.0, 0.0], scale), body([scale * (2.0 - ratio), 0.0, 0.0], scale)]

    def contacts(bodies, velocities=None):
        pair = contact_between(bodies, 0, 1, velocities, threshold=0.0)
        return [] if pair is None else [pair]

    def drive(ratios, object_ids, mu=0.0, speed=0.0):
        log = EventLog()
        topology = SymbolicTopology(log, object_ids, mu=mu)
        for step, ratio in enumerate(ratios):
            bodies = at(ratio)
            velocities = np.zeros((2, 3))
            velocities[1, 1] = speed
            topology.observe(0.01 * (step + 1), bodies, contacts(bodies, velocities), velocities)
        return log.events

    stiffness = np.diag([(2.0 * np.pi * 0.2) ** 2, (2.0 * np.pi * 10.0) ** 2])
    freqs, vectors = modal_analysis(stiffness, np.eye(2), n_modes=2)
    low = modal_participation(vectors[:, 0], vectors, freqs)
    high = modal_participation(vectors[:, 1], vectors, freqs)
    return {
        "hysteresis_chatter": drive([0.0, 0.01, 0.0, 0.015, 0.0], (1, 2)),
        "point_then_release": drive([0.05, -0.05], (1, 2)),
        "area_depth": drive([0.05, 0.25, 0.05], (1, 2)),
        "self_penetration": drive([0.05], (4, 4)),
        "mu0_free": drive([0.05], (1, 2), mu=0.0, speed=1.0),
        "stick_slide": _stick_slide(at, contacts),
        "modal_low_hz_not_excited": low,
        "modal_tracked_mode_excited": high,
    }


def _stick_slide(at, contacts) -> list[dict]:
    log = EventLog()
    topology = SymbolicTopology(log, [1, 2], mu=0.3)
    for step, speed in enumerate((0.0, 5.0e-3, 1.5e-3, 0.0)):
        bodies = at(0.05)
        velocities = np.zeros((2, 3))
        velocities[1, 1] = speed
        topology.observe(0.1 * (step + 1), bodies, contacts(bodies, velocities), velocities)
    return log.events


def observe_graph(topology: SymbolicTopology, graph, time_s: float, gaussians,
                  velocities=None, modes=None, support_force=None) -> list[dict]:
    """Read the graph's current contacts. Does not rebuild them."""
    if graph.full_rebuilds:
        topology.full_rebuilds += graph.full_rebuilds
    return topology.observe(
        time_s, gaussians, graph.active_contacts(), velocities, modes, support_force,
    )
