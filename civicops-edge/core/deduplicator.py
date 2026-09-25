"""
Spatial/temporal de-duplication.

An event is SUPPRESSED if either:
  - it is within DEDUP_COOLDOWN_S seconds of the last emitted event, or
  - it is within DEDUP_RADIUS_M metres of the last emitted location
    (location memory expires after DEDUP_LOCATION_TTL_S so a revisit later counts).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import config
from sensors.gps_node import haversine_m


class Deduplicator:
    def __init__(self, cooldown_s: float | None = None, radius_m: float | None = None,
                 location_ttl_s: float | None = None) -> None:
        self.cooldown_s = config.DEDUP_COOLDOWN_S if cooldown_s is None else cooldown_s
        self.radius_m = config.DEDUP_RADIUS_M if radius_m is None else radius_m
        self.location_ttl_s = config.DEDUP_LOCATION_TTL_S if location_ttl_s is None else location_ttl_s
        self._last_t: Optional[float] = None
        self._last_pos: Optional[Tuple[float, float]] = None
        self._lock = threading.Lock()
        self.suppressed = 0
        self.last_reason = ""

    def check(self, lat: float, lon: float, now: float | None = None) -> bool:
        """Return True if the event should be emitted (and record it). False = duplicate."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._last_t is not None:
                dt = now - self._last_t
                if dt < self.cooldown_s:
                    self.suppressed += 1
                    self.last_reason = f"cooldown ({dt:.2f}s < {self.cooldown_s}s)"
                    return False
                if self._last_pos is not None and dt < self.location_ttl_s:
                    dist = haversine_m(lat, lon, *self._last_pos)
                    if dist < self.radius_m:
                        self.suppressed += 1
                        self.last_reason = f"radius ({dist:.1f}m < {self.radius_m}m)"
                        return False
            self._last_t = now
            self._last_pos = (lat, lon)
            self.last_reason = ""
            return True

    def reset(self) -> None:
        with self._lock:
            self._last_t = None
            self._last_pos = None
