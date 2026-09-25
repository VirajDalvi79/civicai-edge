"""
MPU-6050 accelerometer on I2C-1 (0x68). Samples Z-axis g at ~100 Hz into a
ring buffer; peak_z() returns the strongest vertical hit in a time window.

If the chip is not connected (IMU_MODE=auto), the module disables itself and
peak_z() returns None -> payload kinetics.z_impact_g = null.
"""
from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import config
from sensors.mock_road import get_road

log = logging.getLogger("imu")

# MPU-6050 registers
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
ACCEL_CONFIG = 0x1C
ACCEL_ZOUT_H = 0x3F
WHO_AM_I = 0x75

_RANGE_BITS = {2: 0x00, 4: 0x08, 8: 0x10, 16: 0x18}
_LSB_PER_G = {2: 16384.0, 4: 8192.0, 8: 4096.0, 16: 2048.0}


class IMU:
    def __init__(self, mode: str | None = None) -> None:
        self.requested_mode = config.resolve_mode(mode or config.IMU_MODE)
        self.backend = "none"
        self._bus = None
        self._rng = random.Random(config.MOCK_SEED + 2)
        self._range = config.IMU_ACCEL_RANGE_G if config.IMU_ACCEL_RANGE_G in _RANGE_BITS else 8
        self._buf: Deque[Tuple[float, float]] = deque(
            maxlen=int(config.IMU_BUFFER_S * config.IMU_SAMPLE_HZ) + 1)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples = 0
        self.read_errors = 0

    # -- backend ------------------------------------------------------------
    def _open_hardware(self) -> None:
        from smbus2 import SMBus  # type: ignore
        bus = SMBus(config.IMU_I2C_BUS)
        addr = config.IMU_I2C_ADDRESS
        who = bus.read_byte_data(addr, WHO_AM_I)
        if who not in (0x68, 0x70, 0x72, 0x98):   # MPU-6050 / clones
            log.warning("Unexpected WHO_AM_I 0x%02X - continuing anyway", who)
        bus.write_byte_data(addr, PWR_MGMT_1, 0x00)          # wake up
        time.sleep(0.05)
        bus.write_byte_data(addr, SMPLRT_DIV, 0x07)          # 1 kHz / (1+7) = 125 Hz internal
        bus.write_byte_data(addr, CONFIG, 0x03)              # DLPF ~44 Hz: kills engine buzz
        bus.write_byte_data(addr, ACCEL_CONFIG, _RANGE_BITS[self._range])
        self._bus = bus

    def open(self) -> None:
        mode = self.requested_mode
        if mode == "disabled":
            self.backend = "disabled"
        elif mode == "mock":
            self.backend = "mock"
        else:
            try:
                self._open_hardware()
                self.backend = "mpu6050"
            except Exception as e:
                if mode == "hardware":
                    raise
                log.warning("MPU-6050 not found (%s) - IMU DISABLED, z_impact_g will be null", e)
                self.backend = "disabled"
        log.info("IMU backend: %s (range ±%dg)", self.backend, self._range)

    def _read_z(self) -> Optional[float]:
        if self.backend == "mpu6050":
            try:
                hi, lo = self._bus.read_i2c_block_data(config.IMU_I2C_ADDRESS, ACCEL_ZOUT_H, 2)
                raw = (hi << 8) | lo
                if raw >= 0x8000:
                    raw -= 0x10000
                return raw / _LSB_PER_G[self._range]
            except Exception:
                self.read_errors += 1
                return None
        if self.backend == "mock":
            st = get_road().state()
            z = 1.0 + self._rng.gauss(0, 0.06)                 # road vibration
            if self._rng.random() < 0.004:                     # random small bump
                z += self._rng.uniform(0.2, 0.4)
            if st.active and 0.25 <= st.progress <= 0.35:      # wheel drops into pothole
                z += (st.impact_g - 1.0) * self._rng.uniform(0.85, 1.0)
            return z
        return None

    def _loop(self) -> None:
        period = 1.0 / config.IMU_SAMPLE_HZ
        next_t = time.monotonic()
        while not self._stop.is_set():
            t = time.monotonic()
            z = self._read_z()
            if z is not None:
                with self._lock:
                    self._buf.append((t, z))
                self.samples += 1
            next_t += period
            delay = next_t - time.monotonic()
            if delay < -period * 5:        # fell badly behind; resync instead of bursting
                next_t = time.monotonic()
                delay = 0
            self._stop.wait(max(0.0, delay))

    # -- public API -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.backend in ("mpu6050", "mock")

    def start(self) -> "IMU":
        if self.backend == "none":
            self.open()
        if self.enabled:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="imu", daemon=True)
            self._thread.start()
        return self

    def peak_z(self, window_s: float | None = None, now: float | None = None) -> Optional[float]:
        """Peak |Z| acceleration (g) in the last window_s seconds."""
        if not self.enabled:
            return None
        window_s = window_s or config.FUSION_WINDOW_S
        now = time.monotonic() if now is None else now
        with self._lock:
            vals = [abs(z) for (t, z) in self._buf if now - window_s <= t <= now]
        return round(max(vals), 3) if vals else None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
            self._bus = None
        log.info("IMU stopped")
