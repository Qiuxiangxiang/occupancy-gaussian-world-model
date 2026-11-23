"""Event-driven incremental contact graph.

The hot path updates dirty bodies (added, removed, moved, or scaled) and the
candidate edges incident to them. It does not rebuild every pair and it does
not run on a read. ``oracle_all_pairs`` and ``oracle_full_rebuild`` exist for
tests.

Finite-support policy
---------------------
Phase 0's predicate is the closed-form 1-sigma test in ``contact_between``.
The gaussian density is nonzero everywhere; that tail is not a contact. A
pair can be a contact only when the center distance is below the sum of the
projected radii, and each projected radius is at most ``max(scale)``. The
query ball is that max scale, stretched when ``threshold < 0``, plus a
relative margin of 1e-7 and an absolute margin of 1e-9 length units. A body
whose query ball is larger than four cell widths is kept in an explicit large
set and tested against the dirty body directly. That fallback is linear in
the number of bodies for that body. This is not a worst-case subquadratic
algorithm, and a grid rejection is exact only relative to this predicate.

Motion safety
-------------
If a committed body moves farther than ``motion_slack_factor`` times its
query radius, the commit emits ``motion_safety``. If the segment from the
last committed pose to the new pose comes inside another body's query ball
while the endpoint does not, the commit emits ``possible_tunnel``. The active
contact set is still the endpoint set, so it stays comparable to the all-pairs
oracle. The tunnel scan uses other bodies' last committed poses and runs only
after the slack is exceeded.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from compiler.events import EventLog
from occupancy_gaussian_framework import ContactPair, contact_between, contact_pairs


class _Body:
    def __init__(self, pos, scale, radius):
        self.pos = pos
        self.scale = scale
        self.radius = radius
        self.cells: tuple = ()
        self.in_grid = False
        self.large = False
        self.removed = False


class IncrementalContactGraph:
    def __init__(self, threshold: float = 0.0, support_factor: float = 1.0,
                 rel_margin: float = 1e-7, abs_margin: float = 1e-9,
                 motion_slack_factor: float = 0.25, cell_size: float | None = None,
                 log: EventLog | None = None):
        if support_factor < 1.0:
            raise ValueError(
                "support_factor < 1 can drop a true 1-sigma contact; "
                "the predicate's projected radius reaches max(scale)"
            )
        self.threshold = float(threshold)
        self.support_factor = float(support_factor)
        self.rel_margin = float(rel_margin)
        self.abs_margin = float(abs_margin)
        self.motion_slack_factor = float(motion_slack_factor)
        self.h = None if cell_size is None else float(cell_size)
        if self.h is not None and (not np.isfinite(self.h) or self.h <= 0.0):
            raise ValueError("cell_size must be finite and positive")
        self.log = log if log is not None else EventLog()
        self.bodies: dict[int, _Body] = {}
        self.grid: dict[tuple, set] = defaultdict(set)
        self.large: set[int] = set()
        self.contacts: dict[tuple[int, int], ContactPair] = {}
        self.dirty: set[int] = set()
        self.narrowphase_checks = 0
        self.last_checked_pairs: list[tuple[int, int]] = []
        self.full_rebuilds = 0

    def add(self, index: int, gaussian) -> None:
        if index < 0:
            raise ValueError("contact index must be >= 0")
        if index in self.bodies:
            raise ValueError(f"index {index} is already in the contact graph")
        pos = np.array(gaussian.pos, dtype=np.float64, copy=True)
        scale = np.array(gaussian.scale, dtype=np.float64, copy=True)
        radius = self._radius(scale)
        self._ensure_h(radius)
        self.bodies[index] = _Body(pos, scale, radius)
        self.dirty.add(index)

    def remove(self, index: int, time_s: float = 0.0) -> list[dict]:
        body = self._live(index)
        events = []
        for key in sorted(k for k in self.contacts if index in k):
            del self.contacts[key]
            events.append(self.log.append(
                time_s, "contact_off", i=key[0], j=key[1], pair_id=f"{key[0]}-{key[1]}",
            ))
        self._erase(index)
        body.removed = True
        self.dirty.discard(index)
        return events

    def mark_moved(self, index: int) -> None:
        self._live(index)
        self.dirty.add(index)

    def commit(self, gaussians, time_s: float, velocities=None) -> list[dict]:
        """Update dirty bodies. A clean graph only refreshes relative velocity."""
        if not self.dirty:
            self.last_checked_pairs = []
            if velocities is not None:
                self._refresh_velocity(velocities)
            return []
        events = []
        dirty = sorted(self.dirty)
        dirty_set = set(dirty)
        staged = {}
        for index in dirty:
            body = self.bodies[index]
            pos = np.array(gaussians[index].pos, dtype=np.float64, copy=True)
            scale = np.array(gaussians[index].scale, dtype=np.float64, copy=True)
            radius = self._radius(scale)
            if body.in_grid:
                disp = float(np.linalg.norm(pos - body.pos))
                limit = self.motion_slack_factor * float(body.radius)
                if disp > limit:
                    events.append(self.log.append(
                        time_s, "motion_safety", i=index,
                        displacement_m=disp, limit_m=limit,
                    ))
                    events.extend(self._tunnels(index, body.pos, pos, radius, time_s))
            staged[index] = (pos, scale, radius)

        before = {
            key: self.contacts[key]
            for key in self.contacts
            if key[0] in dirty_set or key[1] in dirty_set
        }
        for key in before:
            del self.contacts[key]
        for index in dirty:
            self._erase(index)
            pos, scale, radius = staged[index]
            body = self.bodies[index]
            body.pos = pos
            body.scale = scale
            body.radius = radius
            self._insert(index)

        checked = []
        found: dict[tuple[int, int], ContactPair | None] = {}
        for index in dirty:
            body = self.bodies[index]
            for other in sorted(self._query(body.pos, body.radius)):
                if other == index or not self.bodies[other].in_grid:
                    continue
                pair_key = (index, other) if index < other else (other, index)
                owner = pair_key[0]
                if owner in dirty_set and index != owner:
                    continue
                if pair_key in found:
                    continue
                if not self._balls_overlap(pair_key[0], pair_key[1]):
                    continue
                self.narrowphase_checks += 1
                checked.append(pair_key)
                found[pair_key] = contact_between(
                    gaussians, pair_key[0], pair_key[1], velocities, self.threshold,
                )
        self.last_checked_pairs = checked
        after = {}
        for key, pair in found.items():
            if pair is not None:
                after[key] = pair
                self.contacts[key] = pair
        for key in sorted(set(after) - set(before)):
            events.append(self.log.append(
                time_s, "contact_on", i=key[0], j=key[1], pair_id=f"{key[0]}-{key[1]}",
            ))
        for key in sorted(set(before) - set(after)):
            events.append(self.log.append(
                time_s, "contact_off", i=key[0], j=key[1], pair_id=f"{key[0]}-{key[1]}",
            ))
        self.dirty.clear()
        return events

    def _balls_overlap(self, i: int, j: int) -> bool:
        left, right = self.bodies[i], self.bodies[j]
        return float(np.linalg.norm(left.pos - right.pos)) <= left.radius + right.radius

    def active_contacts(self) -> list[ContactPair]:
        if self.dirty:
            raise RuntimeError(
                "contact graph is dirty; commit moved, scaled, added, or "
                "removed bodies before reading contacts"
            )
        return [self.contacts[key] for key in sorted(self.contacts)]

    def _radius(self, scale: np.ndarray) -> float:
        smax = float(np.max(scale))
        if not np.isfinite(smax) or smax < 0.0:
            raise ValueError("scale must be finite and non-negative")
        stretch = 1.0 + max(-self.threshold, 0.0)
        return smax * self.support_factor * stretch * (1.0 + self.rel_margin) + self.abs_margin

    def _ensure_h(self, radius: float) -> None:
        if self.h is None:
            self.h = max(2.0 * radius, 1e-4)

    def _live(self, index: int) -> _Body:
        body = self.bodies.get(index)
        if body is None or body.removed:
            raise KeyError(f"index {index} is not an active contact-graph body")
        return body

    def _erase(self, index: int) -> None:
        body = self.bodies[index]
        if body.large:
            self.large.discard(index)
        for cell in body.cells:
            bucket = self.grid.get(cell)
            if bucket is None:
                continue
            bucket.discard(index)
            if not bucket:
                del self.grid[cell]
        body.cells = ()
        body.in_grid = False
        body.large = False

    def _bounds(self, pos, radius):
        lo = np.floor((pos - radius) / self.h).astype(np.int64)
        hi = np.floor((pos + radius) / self.h).astype(np.int64)
        return lo, hi

    def _insert(self, index: int) -> None:
        body = self.bodies[index]
        if body.radius > 4.0 * self.h:
            body.large = True
            body.cells = ()
            self.large.add(index)
            body.in_grid = True
            return
        lo, hi = self._bounds(body.pos, body.radius)
        cells = []
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    cell = (ix, iy, iz)
                    self.grid[cell].add(index)
                    cells.append(cell)
        body.cells = tuple(cells)
        body.large = False
        body.in_grid = True

    def _query(self, pos, radius) -> set[int]:
        found = {i for i in self.large if self.bodies[i].in_grid and not self.bodies[i].removed}
        if radius > 4.0 * self.h:
            found.update(
                i for i, body in self.bodies.items() if body.in_grid and not body.removed
            )
            return found
        lo, hi = self._bounds(pos, radius)
        for ix in range(int(lo[0]), int(hi[0]) + 1):
            for iy in range(int(lo[1]), int(hi[1]) + 1):
                for iz in range(int(lo[2]), int(hi[2]) + 1):
                    found.update(self.grid.get((ix, iy, iz), ()))
        return found

    def _refresh_velocity(self, velocities) -> None:
        for key, pair in list(self.contacts.items()):
            i, j = key
            rel = np.asarray(velocities[j] - velocities[i], dtype=np.float64)
            self.contacts[key] = ContactPair(i, j, pair.normal, pair.penetration, rel)

    def _tunnels(self, index, start, end, radius, time_s) -> list[dict]:
        events = []
        for other_index in sorted(self.bodies):
            other = self.bodies[other_index]
            if other_index == index or not other.in_grid or other.removed:
                continue
            segment = _segment_distance(other.pos, start, end)
            endpoint = float(np.linalg.norm(other.pos - end))
            reach = radius + other.radius
            if segment <= reach and endpoint > reach:
                events.append(self.log.append(
                    time_s, "possible_tunnel", i=index, j=other_index,
                    segment_distance_m=segment, endpoint_distance_m=endpoint,
                ))
        return events


def _segment_distance(point, start, end) -> float:
    chord = end - start
    length2 = float(np.dot(chord, chord))
    if length2 <= 1e-24:
        return float(np.linalg.norm(point - start))
    t = float(np.dot(point - start, chord) / length2)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(point - (start + t * chord)))


def oracle_all_pairs(gaussians, velocities=None, threshold: float = 0.0):
    """All-pairs reference. Tests only."""
    return contact_pairs(gaussians, velocities=velocities, threshold=threshold)


def oracle_full_rebuild(gaussians, threshold: float = 0.0, velocities=None, **kwargs):
    """Build a fresh graph from every body. Tests only; not the hot path."""
    graph = IncrementalContactGraph(threshold=threshold, **kwargs)
    graph.full_rebuilds = 1
    for index, gaussian in enumerate(gaussians):
        graph.add(index, gaussian)
    graph.commit(gaussians, 0.0, velocities)
    return graph
