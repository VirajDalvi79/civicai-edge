"""
u-blox NEO-6M GPS over UART (/dev/serial0 @ 9600) with a synthetic fallback.

- Parses $xxRMC and $xxGGA sentences (GP/GN/GL prefixes), validates checksums.
- If there is no valid fix for GPS_FIX_TIMEOUT_S (cold start indoors can take
  minutes), positions come from a simulated drive along Mira Road -> WEH and
  are flagged is_mock = true.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import config

log = logging.getLogger("gps")


# ---------------------------------------------------------------------------
# NMEA parsing (no external dependency)
# ---------------------------------------------------------------------------
def nmea_checksum_ok(sentence: str) -> bool:
    sentence = sentence.strip()
    if not sentence.startswith("$") or "*" not in sentence:
        return False
    body, _, given = sentence[1:].partition("*")
    calc = 0
    for ch in body:
        calc ^= ord(ch)
    try:
        return calc == int(given[:2], 16)
    except ValueError:
        return False


def _dm_to_deg(value: str, hemi: str) -> Optional[float]:
    """NMEA ddmm.mmmm / dddmm.mmmm -> signed decimal degrees."""
    if not value or not hemi:
        return None
    try:
        dot = value.index(".")
        deg = float(value[:dot - 2])
        minutes = float(value[dot - 2:])
    except ValueError:
        return None
    out = deg + minutes / 60.0
    return -out if hemi in ("S", "W") else out


@dataclass
class Fix:
    latitude: float
    longitude: float
    is_mock: bool
    source: str = ""           # "RMC" | "GGA" | "synthetic"
    satellites: Optional[int] = None
    hdop: Optional[float] = None
    speed_mps: Optional[float] = None


def parse_nmea(sentence: str) -> Optional[Fix]:
    """Return a Fix for a valid RMC/GGA sentence with a position fix, else None."""
    if not nmea_checksum_ok(sentence):
        return None
    fields = sentence.strip()[1:].split("*")[0].split(",")
    kind = fields[0][-3:]
    try:
        if kind == "RMC" and len(fields) >= 7:
            if fields[2] != "A":                 # V = void (no fix)
                return None
            lat = _dm_to_deg(fields[3], fields[4])
            lon = _dm_to_deg(fields[5], fields[6])
            spd = float(fields[7]) * 0.514444 if len(fields) > 7 and fields[7] else None
            if lat is None or lon is None:
                return None
            return Fix(lat, lon, False, "RMC", speed_mps=spd)
        if kind == "GGA" and len(fields) >= 9:
            if not fields[6] or int(fields[6]) == 0:   # fix quality 0 = invalid
                return None
            lat = _dm_to_deg(fields[2], fields[3])
            lon = _dm_to_deg(fields[4], fields[5])
            if lat is None or lon is None:
                return None
            sats = int(fields[7]) if fields[7] else None
            hdop = float(fields[8]) if fields[8] else None
            return Fix(lat, lon, False, "GGA", satellites=sats, hdop=hdop)
    except (ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Geo helpers
# ---------------------------------------------------------------------------
def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


class SyntheticRoute:
    """Drives back and forth along a polyline at constant speed."""

    def __init__(self, points: List[Tuple[float, float]], speed_mps: float) -> None:
        self.points = points
        self.speed = speed_mps
        self.seg_len = [haversine_m(*points[i], *points[i + 1]) for i in range(len(points) - 1)]
        self.total = sum(self.seg_len)
        self.t0 = time.monotonic()

    def position(self, now: Optional[float] = None) -> Tuple[float, float]:
        now = time.monotonic() if now is None else now
        dist = (now - self.t0) * self.speed
        lap = dist % (2 * self.total)
        d = lap if lap <= self.total else 2 * self.total - lap   # ping-pong
        for i, L in enumerate(self.seg_len):
            if d <= L or i == len(self.seg_len) - 1:
                f = 0.0 if L == 0 else min(1.0, d / L)
                (a_lat, a_lon), (b_lat, b_lon) = self.points[i], self.points[i + 1]
                return a_lat + (b_lat - a_lat) * f, a_lon + (b_lon - a_lon) * f
            d -= L
        return self.points[-1]


# ---------------------------------------------------------------------------
# GPS node
# ---------------------------------------------------------------------------
class GPSNode:
    def __init__(self, mode: str | None = None) -> None:
        self.requested_mode = config.resolve_mode(mode or config.GPS_MODE)
        self.backend = "none"
        self._ser = None
        self._route = SyntheticRoute(config.GPS_MOCK_ROUTE, config.GPS_MOCK_SPEED_MPS)
        self._lock = threading.Lock()
        self._last_fix: Optional[Fix] = None
        self._last_fix_t = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.sentences = 0
        self.bad_checksums = 0
        self._had_fix = False

    def open(self) -> None:
        mode = self.requested_mode
        if mode in ("mock", "disabled"):
            self.backend = "synthetic"
        else:
            try:
                import serial  # type: ignore
                self._ser = serial.Serial(config.GPS_PORT, config.GPS_BAUD, timeout=1.0)
                self.backend = "serial"
            except Exception as e:
                if mode == "hardware":
                    raise
                log.warning("GPS serial %s unavailable (%s) - using synthetic route", config.GPS_PORT, e)
                self.backend = "synthetic"
        log.info("GPS backend: %s", self.backend)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ser.readline()
            except Exception as e:
                log.error("GPS serial read error: %s", e)
                self._stop.wait(1.0)
                continue
            if not raw:
                continue
            line = raw.decode("ascii", errors="ignore").strip()
            if not line.startswith("$"):
                continue
            self.sentences += 1
            if not nmea_checksum_ok(line):
                self.bad_checksums += 1
                continue
            fix = parse_nmea(line)
            if fix:
                with self._lock:
                    self._last_fix, self._last_fix_t = fix, time.monotonic()
                if not self._had_fix:
                    log.info("GPS fix acquired: %.6f, %.6f", fix.latitude, fix.longitude)
                    self._had_fix = True

    def start(self) -> "GPSNode":
        if self.backend == "none":
            self.open()
        if self.backend == "serial":
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="gps", daemon=True)
            self._thread.start()
        return self

    def has_fix(self) -> bool:
        with self._lock:
            return (self._last_fix is not None and
                    time.monotonic() - self._last_fix_t <= config.GPS_FIX_TIMEOUT_S)

    def position(self) -> Fix:
        """Real fix if fresh, else synthetic Mumbai coordinates (is_mock=True)."""
        with self._lock:
            fix, t = self._last_fix, self._last_fix_t
        if fix is not None and time.monotonic() - t <= config.GPS_FIX_TIMEOUT_S:
            return fix
        if self._had_fix and fix is not None:
            log.debug("GPS fix stale (%.1fs) - synthetic", time.monotonic() - t)
        lat, lon = self._route.position()
        return Fix(round(lat, 7), round(lon, 7), True, "synthetic")

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        log.info("GPS stopped")
