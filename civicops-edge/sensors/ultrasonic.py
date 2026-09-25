"""
HC-SR04 ultrasonic depth sensor (mounted pointing DOWN at the road).

  depth_drop_cm = current_distance - flat_road_baseline
A pothole makes the road surface farther away -> positive depth.

WIRING WARNING: ECHO outputs 5 V. It MUST go through a 1k/2k voltage divider
before GPIO24, or you risk destroying the Pi's GPIO.
"""
from __future__ import annotations

import json
import logging
import random
import statistics
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import config
from sensors.mock_road import get_road

log = logging.getLogger("ultrasonic")


class Ultrasonic:
    def __init__(self, mode: str | None = None) -> None:
        self.requested_mode = config.resolve_mode(mode or config.ULTRASONIC_MODE)
        self.backend = "none"
        self._dev = None
        self._rng = random.Random(config.MOCK_SEED + 1)
        self._buf: Deque[Tuple[float, float]] = deque(
            maxlen=int(config.ULTRASONIC_BUFFER_S * config.ULTRASONIC_SAMPLE_HZ) + 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.baseline_cm = self._load_baseline()
        self.timeouts = 0
        self.samples = 0

    # -- baseline -----------------------------------------------------------
    def _load_baseline(self) -> float:
        path = config.ULTRASONIC_BASELINE_FILE
        try:
            if path.exists():
                data = json.loads(path.read_text())
                b = float(data["baseline_cm"])
                log.info("Loaded ultrasonic baseline %.1f cm from %s", b, path)
                return b
        except Exception as e:
            log.warning("Bad baseline file %s (%s); using default", path, e)
        return config.ULTRASONIC_DEFAULT_BASELINE_CM

    def calibrate(self, seconds: float | None = None, save: bool = True) -> float:
        """Park on flat road, then call this. Median of samples becomes the baseline."""
        seconds = seconds or config.ULTRASONIC_CALIBRATION_S
        log.info("Calibrating baseline for %.1f s - keep the sensor over FLAT road", seconds)
        vals = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            d = self._measure()
            if d is not None:
                vals.append(d)
            time.sleep(1.0 / config.ULTRASONIC_SAMPLE_HZ)
        if len(vals) < 5:
            raise RuntimeError(f"Calibration failed: only {len(vals)} valid readings")
        self.baseline_cm = round(statistics.median(vals), 2)
        spread = statistics.pstdev(vals)
        log.info("Baseline = %.2f cm (std %.2f, n=%d)", self.baseline_cm, spread, len(vals))
        if save:
            config.ULTRASONIC_BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
            config.ULTRASONIC_BASELINE_FILE.write_text(json.dumps(
                {"baseline_cm": self.baseline_cm, "std_cm": round(spread, 3), "n": len(vals),
                 "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
        return self.baseline_cm

    # -- backend ------------------------------------------------------------
    def open(self) -> None:
        mode = self.requested_mode
        if mode == "disabled":
            self.backend = "disabled"
        elif mode == "mock":
            self.backend = "mock"
        else:
            try:
                from gpiozero import DistanceSensor  # type: ignore
                self._dev = DistanceSensor(
                    echo=config.ULTRASONIC_ECHO_PIN,
                    trigger=config.ULTRASONIC_TRIG_PIN,
                    max_distance=config.ULTRASONIC_MAX_DISTANCE_CM / 100.0,
                    queue_len=3,          # light smoothing, low latency
                )
                self.backend = "gpiozero"
            except Exception as e:
                if mode == "hardware":
                    raise
                log.warning("HC-SR04 unavailable (%s) - ultrasonic DISABLED", e)
                self.backend = "disabled"
        log.info("Ultrasonic backend: %s (baseline %.1f cm)", self.backend, self.baseline_cm)

    def _measure(self) -> Optional[float]:
        """One distance reading in cm, or None if invalid / timed out."""
        if self.backend == "gpiozero":
            try:
                d = self._dev.distance * 100.0
            except Exception as e:
                log.debug("read error %s", e)
                return None
            # gpiozero reports max_distance when the echo times out
            if d >= config.ULTRASONIC_MAX_DISTANCE_CM - 0.5 or d < config.ULTRASONIC_MIN_DISTANCE_CM:
                self.timeouts += 1
                return None
            return d
        if self.backend == "mock":
            st = get_road().state()
            d = self.baseline_cm + self._rng.gauss(0, 0.3)
            # sensor (under the bumper) crosses the pothole shortly after the camera sees it
            if st.active and 0.15 <= st.progress <= 0.45:
                d += st.depth_cm * (1 - abs(st.progress - 0.30) / 0.15 * 0.5)
            if self._rng.random() < 0.01:          # occasional dropout, like the real thing
                self.timeouts += 1
                return None
            return d
        return None

    def _loop(self) -> None:
        period = 1.0 / config.ULTRASONIC_SAMPLE_HZ
        while not self._stop.is_set():
            t = time.monotonic()
            d = self._measure()
            if d is not None:
                with self._lock:
                    self._buf.append((t, d))
                self.samples += 1
            self._stop.wait(max(0.0, period - (time.monotonic() - t)))

    # -- public API -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.backend in ("gpiozero", "mock")

    def start(self) -> "Ultrasonic":
        if self.backend == "none":
            self.open()
        if self.enabled:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="ultrasonic", daemon=True)
            self._thread.start()
        return self

    def latest_distance(self) -> Optional[float]:
        with self._lock:
            return self._buf[-1][1] if self._buf else None

    def max_depth(self, window_s: float | None = None, now: float | None = None) -> Optional[float]:
        """Largest depth drop (cm) below baseline in the last window_s seconds.
        Returns None if the sensor is off or has no valid samples in the window."""
        if not self.enabled:
            return None
        window_s = window_s or config.FUSION_WINDOW_S
        now = time.monotonic() if now is None else now
        with self._lock:
            vals = [d for (t, d) in self._buf if now - window_s <= t <= now]
        if not vals:
            return None
        drop = max(vals) - self.baseline_cm
        return round(drop, 2) if drop >= config.DEPTH_NOISE_FLOOR_CM else 0.0

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._dev is not None:
            try:
                self._dev.close()   # releases GPIO23/24
            except Exception as e:
                log.warning("GPIO close error: %s", e)
            self._dev = None
        log.info("Ultrasonic stopped")
