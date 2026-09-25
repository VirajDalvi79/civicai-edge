"""
Shared, deterministic "virtual road" used by every mock sensor.

All mock modules ask this timeline whether the vehicle is currently passing a
pothole, so the synthetic camera frame, ultrasonic depth dip and IMU spike line
up in time - exactly like real hardware would - and fusion can be tested
end-to-end without a Pi.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass

import config


@dataclass(frozen=True)
class PotholeState:
    active: bool
    index: int          # which pothole on the virtual road
    progress: float     # 0..1 while passing over it
    depth_cm: float     # true depth of this pothole
    width_frac: float   # pothole size relative to frame width
    impact_g: float     # peak vertical g when a wheel hits it


class MockRoad:
    def __init__(self, period_s: float | None = None, duration_s: float | None = None,
                 seed: int | None = None) -> None:
        self.period_s = period_s or config.MOCK_POTHOLE_PERIOD_S
        self.duration_s = duration_s or config.MOCK_POTHOLE_DURATION_S
        self.seed = config.MOCK_SEED if seed is None else seed
        # First pothole appears ~1 s after start so short test runs see one.
        self.t0 = time.monotonic() - (self.period_s - 1.0)

    def _params(self, index: int) -> tuple[float, float, float]:
        rng = random.Random(self.seed * 100_003 + index)
        depth = rng.uniform(2.5, 12.0)
        width = rng.uniform(0.15, 0.40)
        impact = 1.0 + 0.12 * depth + rng.uniform(-0.1, 0.3)  # deeper -> harder hit
        return depth, width, impact

    def state(self, now: float | None = None) -> PotholeState:
        now = time.monotonic() if now is None else now
        elapsed = now - self.t0
        index = int(elapsed // self.period_s)
        phase = elapsed - index * self.period_s
        depth, width, impact = self._params(index)
        active = phase < self.duration_s
        progress = phase / self.duration_s if active else 0.0
        return PotholeState(active, index, progress, depth, width, impact)


_ROAD: MockRoad | None = None


def get_road() -> MockRoad:
    """Process-wide singleton so all mock sensors share one timeline."""
    global _ROAD
    if _ROAD is None:
        _ROAD = MockRoad()
    return _ROAD


def reset_road(**kwargs) -> MockRoad:
    global _ROAD
    _ROAD = MockRoad(**kwargs)
    return _ROAD
