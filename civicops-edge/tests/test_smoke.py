"""
Smoke + integration tests. Run on any machine (no Pi needed):

    MOCK_HARDWARE=1 python -m pytest -v tests/
"""
from __future__ import annotations

import http.server
import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

# ---- environment must be set BEFORE config is imported -----------------
_TMP = Path(tempfile.mkdtemp(prefix="civicops-test-"))
os.environ["MOCK_HARDWARE"] = "1"
os.environ["OUTPUT_DIR"] = str(_TMP / "output")
os.environ["ULTRASONIC_BASELINE_FILE"] = str(_TMP / "calibration" / "baseline.json")
os.environ.setdefault("MOCK_POTHOLE_PERIOD_S", "3.5")
os.environ["STATS_INTERVAL_S"] = "2"
os.environ["TELEMETRY_URL"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import config  # noqa: E402
from core.deduplicator import Deduplicator  # noqa: E402
from core.dispatcher import TelemetryDispatcher  # noqa: E402
from core.schema import (GPSInfo, KineticsInfo, SurfaceInfo, TelemetryPayload,  # noqa: E402
                         VisionInfo, classify_severity, utc_now_iso, validate_payload)
from model.detector import Detector  # noqa: E402
from sensors.camera import Camera, MockFrameSource, clahe_gray, gray_world, preprocess  # noqa: E402
from sensors.gps_node import GPSNode, haversine_m, nmea_checksum_ok, parse_nmea  # noqa: E402
from sensors.imu import IMU  # noqa: E402
from sensors.mock_road import get_road, reset_road  # noqa: E402
from sensors.ultrasonic import Ultrasonic  # noqa: E402


def _force_road(progress: float):
    """Put the shared mock road at a given point of a pothole pass (or off-pothole if None)."""
    road = reset_road()
    if progress is None:
        road.t0 = time.monotonic() - (road.period_s + road.duration_s + 0.5)
    else:
        road.t0 = time.monotonic() - (road.period_s + progress * road.duration_s)
    return road


def _nmea(body: str) -> str:
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"${body}*{cs:02X}"


# ---------------------------------------------------------------------------
# Config / preprocessing / camera
# ---------------------------------------------------------------------------
def test_mock_mode_enabled():
    assert config.MOCK_HARDWARE is True
    for m in (config.CAMERA_MODE, config.ULTRASONIC_MODE, config.IMU_MODE, config.GPS_MODE):
        assert config.resolve_mode(m) == "mock"


def test_preprocess_clahe_removes_colour_cast():
    frame = MockFrameSource(320, 240).read()
    out = clahe_gray(frame)
    assert out.shape == frame.shape
    assert np.array_equal(out[:, :, 0], out[:, :, 1]) and np.array_equal(out[:, :, 1], out[:, :, 2])
    assert preprocess(frame, "none") is frame


def test_gray_world_balances_channels():
    frame = MockFrameSource(320, 240).read()          # has a magenta tint
    before = frame.reshape(-1, 3).mean(0)
    after = gray_world(frame).reshape(-1, 3).mean(0)
    assert np.ptp(after) < np.ptp(before)


def test_camera_mock_start_read_stop():
    cam = Camera().start()
    try:
        frame = cam.read(timeout=2)
        assert cam.backend == "mock"
        assert frame is not None and frame.shape == (config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3)
        time.sleep(0.2)
        fid1, _, _ = cam.latest()
        time.sleep(0.2)
        fid2, _, _ = cam.latest()
        assert fid2 > fid1, "capture thread should keep producing frames"
    finally:
        cam.stop()
    assert not cam._thread.is_alive()


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["none", "clahe_gray", "grayworld"])
def test_mock_detector_finds_pothole(mode):
    _force_road(0.5)
    frame = preprocess(MockFrameSource(config.CAMERA_WIDTH, config.CAMERA_HEIGHT).read(), mode)
    dets = Detector().load().detect(frame)
    assert dets, f"expected a detection with preprocess={mode}"
    d = dets[0]
    assert d.confidence > config.DETECTION_CONFIDENCE
    x1, y1, x2, y2 = d.bbox
    assert x2 > x1 and y2 > y1 and d.area_px > 0


def test_mock_detector_no_false_positive_on_clean_road():
    _force_road(None)
    frame = preprocess(MockFrameSource(config.CAMERA_WIDTH, config.CAMERA_HEIGHT).read())
    assert Detector().load().detect(frame) == []


# ---------------------------------------------------------------------------
# Ultrasonic / IMU
# ---------------------------------------------------------------------------
def test_ultrasonic_flat_road_has_no_depth():
    _force_road(None)
    us = Ultrasonic().start()
    try:
        time.sleep(0.6)
        assert us.backend == "mock"
        assert abs(us.latest_distance() - us.baseline_cm) < 2.0
        assert us.max_depth(0.5) == 0.0
    finally:
        us.stop()


def test_ultrasonic_detects_depth_drop():
    road = _force_road(0.30)
    us = Ultrasonic()
    us.open()
    readings = [us._measure() for _ in range(20)]
    d = max(r for r in readings if r is not None)
    assert d - us.baseline_cm > road.state().depth_cm * 0.8


def test_ultrasonic_calibration_saves_file():
    _force_road(None)
    us = Ultrasonic()
    us.open()
    b = us.calibrate(seconds=0.5)
    assert abs(b - config.ULTRASONIC_DEFAULT_BASELINE_CM) < 1.0
    saved = json.loads(config.ULTRASONIC_BASELINE_FILE.read_text())
    assert saved["baseline_cm"] == b
    assert Ultrasonic()._load_baseline() == b


def test_imu_rest_and_impact():
    _force_road(None)
    imu = IMU().start()
    try:
        time.sleep(0.5)
        assert imu.samples > 20                       # ~100 Hz
        rest = imu.peak_z(0.5)
        assert 0.8 < rest < 1.6
    finally:
        imu.stop()
    road = _force_road(0.30)
    imu2 = IMU()
    imu2.open()
    z = max(imu2._read_z() for _ in range(10))
    assert z > 1.0 + (road.state().impact_g - 1.0) * 0.7


def test_missing_hardware_degrades_gracefully(monkeypatch):
    """On a non-Pi with MOCK off, sensors must disable themselves, not crash."""
    monkeypatch.setattr(config, "MOCK_HARDWARE", False)
    monkeypatch.setattr(config, "GPS_PORT", str(_TMP / "no-such-serial"))
    imu = IMU(mode="auto")
    imu.open()
    assert imu.backend == "disabled" and imu.peak_z() is None
    us = Ultrasonic(mode="auto")
    us.open()
    assert us.backend == "disabled" and us.max_depth() is None
    gps = GPSNode(mode="auto").start()
    assert gps.backend == "synthetic" and gps.position().is_mock
    imu.stop(); us.stop(); gps.stop()


# ---------------------------------------------------------------------------
# GPS
# ---------------------------------------------------------------------------
def test_nmea_reference_sentences():
    rmc = "$GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W*6A"
    gga = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*47"
    assert nmea_checksum_ok(rmc) and nmea_checksum_ok(gga)
    f = parse_nmea(rmc)
    assert f and abs(f.latitude - 48.1173) < 1e-4 and abs(f.longitude - 11.516667) < 1e-4
    g = parse_nmea(gga)
    assert g and g.satellites == 8 and g.hdop == 0.9


def test_nmea_mumbai_and_invalid():
    s = _nmea("GNRMC,061500.00,A,1916.872,N,07251.366,E,0.5,,250926,,,A")
    f = parse_nmea(s)
    assert f and abs(f.latitude - 19.2812) < 1e-4 and abs(f.longitude - 72.8561) < 1e-4
    assert not f.is_mock
    assert parse_nmea(s[:-2] + "00") is None                                # bad checksum
    assert parse_nmea(_nmea("GPRMC,061500.00,V,,,,,,,250926,,,N")) is None   # no fix
    assert parse_nmea(_nmea("GPGGA,061500.00,,,,,0,00,99.9,,,,,,")) is None  # quality 0
    assert parse_nmea("garbage") is None


def test_gps_synthetic_fallback_moves_around_mumbai():
    gps = GPSNode().start()
    p1 = gps.position()
    gps._route.t0 -= 10                        # pretend 10 s passed
    p2 = gps.position()
    gps.stop()
    for p in (p1, p2):
        assert p.is_mock
        assert 19.2 < p.latitude < 19.32 and 72.83 < p.longitude < 72.89
    moved = haversine_m(p1.latitude, p1.longitude, p2.latitude, p2.longitude)
    assert 60 < moved < 100                    # 8 m/s * 10 s


# ---------------------------------------------------------------------------
# Dedup / severity / schema
# ---------------------------------------------------------------------------
def test_dedup_cooldown_radius_and_ttl():
    d = Deduplicator(cooldown_s=3.0, radius_m=5.0, location_ttl_s=60)
    lat, lon = 19.2812, 72.8561
    assert d.check(lat, lon, now=100.0)
    assert not d.check(19.29, 72.87, now=101.0)          # far away but within 3 s
    assert "cooldown" in d.last_reason
    assert not d.check(lat + 0.00002, lon, now=105.0)    # ~2 m away after 5 s
    assert "radius" in d.last_reason
    assert d.check(lat + 0.0001, lon, now=106.0)         # ~11 m away, 6 s later
    assert d.check(lat + 0.0001, lon, now=200.0)         # same spot but memory expired
    assert d.suppressed == 2


@pytest.mark.parametrize("depth,g,area,expected", [
    (1.0, 1.1, 1000, "LOW"),
    (None, None, 0, "LOW"),
    (5.0, None, 0, "MEDIUM"),
    (None, 2.0, 0, "MEDIUM"),
    (0.0, 1.0, 40_000, "MEDIUM"),       # big bbox (13% of 640x480)
    (9.0, 1.2, 0, "CRITICAL"),
    (2.0, 3.0, 0, "CRITICAL"),
])
def test_severity(depth, g, area, expected):
    assert classify_severity(depth, g, area, 640 * 480) == expected


def _sample_payload(**over) -> dict:
    p = TelemetryPayload(
        node_id=config.NODE_ID, timestamp=utc_now_iso(),
        gps=GPSInfo(19.2812, 72.8561, True),
        vision=VisionInfo(True, 0.81, [10, 20, 110, 90], 7000),
        kinetics=KineticsInfo(1.9), surface=SurfaceInfo(6.2, 30.0),
        severity_level="MEDIUM").to_dict()
    p.update(over)
    return p


def test_payload_schema_exact_shape():
    p = _sample_payload()
    validate_payload(p)
    assert list(p) == ["node_id", "timestamp", "gps", "vision", "kinetics", "surface", "severity_level"]
    assert set(p["gps"]) == {"latitude", "longitude", "is_mock"}
    assert set(p["vision"]) == {"detected", "confidence", "bbox", "area_px"}
    assert p["kinetics"] == {"z_impact_g": 1.9, "baseline_g": 1.0}
    assert set(p["surface"]) == {"depth_cm", "baseline_distance_cm"}
    json.loads(json.dumps(p))
    with pytest.raises(ValueError):
        validate_payload(_sample_payload(severity_level="HIGH"))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
class _Collector(http.server.BaseHTTPRequestHandler):
    received: list = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        _Collector.received.append(json.loads(body))
        self.send_response(201)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_dispatcher_writes_jsonl_and_posts():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    path = _TMP / "d1" / "events.jsonl"
    disp = TelemetryDispatcher(path, url=f"http://127.0.0.1:{srv.server_port}/events").start()
    for _ in range(3):
        disp.submit(_sample_payload())
    disp.stop()
    srv.shutdown()
    lines = path.read_text().splitlines()
    assert len(lines) == 3 and disp.posted == 3
    assert len(_Collector.received) == 3
    validate_payload(_Collector.received[0])


def test_dispatcher_spools_when_offline(monkeypatch):
    monkeypatch.setattr(config, "TELEMETRY_MAX_RETRIES", 1)
    path = _TMP / "d2" / "events.jsonl"
    disp = TelemetryDispatcher(path, url="http://127.0.0.1:9/unreachable").start()
    disp.submit(_sample_payload())
    disp.stop()
    assert len(path.read_text().splitlines()) == 1
    assert disp.failed == 1
    assert len((path.parent / "unsent.jsonl").read_text().splitlines()) == 1


# ---------------------------------------------------------------------------
# End-to-end daemon
# ---------------------------------------------------------------------------
def _new_daemon(duration):
    import main
    reset_road()
    return main.EdgeDaemon(duration=duration)


def test_end_to_end_fused_events():
    d = _new_daemon(duration=8.5)      # potholes at ~1 s, ~4.5 s, ~8 s
    d.run()
    assert d.frames_processed > 50
    fused = [e for e in d.events if e["vision"]["detected"]]
    assert len(fused) >= 2, f"expected >=2 vision events, got {d.events}"
    for e in d.events:
        validate_payload(e)
    for e in fused:
        assert e["vision"]["confidence"] > config.DETECTION_CONFIDENCE
        assert e["surface"]["depth_cm"] is not None and e["surface"]["depth_cm"] > 1.0, e
        assert e["kinetics"]["z_impact_g"] is not None and e["kinetics"]["z_impact_g"] > 1.0
        assert e["gps"]["is_mock"] is True
    # JSONL on disk matches in-memory events
    lines = [json.loads(x) for x in config.TELEMETRY_JSONL.read_text().splitlines()]
    assert lines[-len(d.events):] == d.events
    # no two events inside the 3 s cooldown (one pothole -> one event)
    from datetime import datetime
    ts = [datetime.fromisoformat(e["timestamp"]) for e in d.events]
    assert all((b - a).total_seconds() >= config.DEDUP_COOLDOWN_S - 0.05 for a, b in zip(ts, ts[1:]))
    # sensor threads & camera released
    assert not d.camera._thread.is_alive()
    assert not d.imu._thread.is_alive() and not d.ultrasonic._thread.is_alive()


def test_sigint_graceful_shutdown():
    d = _new_daemon(duration=30)
    d.install_signal_handlers()
    timer = threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT))
    timer.start()
    t = time.monotonic()
    try:
        d.run()
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
    assert time.monotonic() - t < 5, "SIGINT should stop the daemon promptly"
    assert d.stop_event.is_set() and d._shut_down
    assert not d.camera._thread.is_alive()
