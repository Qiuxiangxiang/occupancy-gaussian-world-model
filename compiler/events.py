"""Chronological event log shared by the contact graph and the M1 harness.

M1 records contact-topology changes and modal spectra. It does not classify
normal/tangential/modal-participation axes (M3) and it does not track
eigenmode identity across permutations (M2).
"""
from __future__ import annotations

import math

import numpy as np


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"event field is not finite: {value}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"event field is not JSON-able: {type(value).__name__}")


class EventLog:
    """Append-only log. ``seq`` is the append order and is the replay key."""

    def __init__(self):
        self.events: list[dict] = []

    def append(self, time_s: float, kind: str, **payload) -> dict:
        event = {
            "seq": len(self.events),
            "time_s": float(time_s),
            "kind": str(kind),
        }
        for key, value in payload.items():
            event[str(key)] = _jsonable(value)
        self.events.append(event)
        return event

    def kinds(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["kind"] == kind]
