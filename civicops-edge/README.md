# CivicOps Edge: Phase 1 sensing & detection daemon

Raspberry Pi 4 daemon that spots potholes with YOLOv8n (NCNN), fuses the detection with HC-SR04 depth, MPU-6050 impact g and NEO-6M GPS, de-duplicates the result, and emits JSON telemetry.

**New Pi? Start with [docs/PI_SETUP_GUIDE.md](docs/PI_SETUP_GUIDE.md).**

## Quick start (any laptop, no hardware)
```bash
pip install -r requirements.txt          # or just: opencv-python-headless numpy pyserial requests pytest
MOCK_HARDWARE=1 python -m pytest -q tests/
python main.py --mock --duration 30
```

## Layout
```
config.py              every setting; override any of them with an env var of the same name
main.py                orchestrator (threads + fusion + dedup + dispatch), --mock / --calibrate / --duration
hw_check.py            per-sensor bring-up tool for the Pi
desktop_receiver.py    stdlib HTTP receiver to watch events on your PC
sensors/               camera (picamera2 → OpenCV → mock, NoIR handling), ultrasonic, imu, gps_node, mock_road
model/                 detector (NCNN → ONNX → .pt), download_assets (export / --pothole / benchmark)
core/                  schema, deduplicator, dispatcher (JSONL + HTTP POST with offline spool)
deploy/                systemd unit
tests/test_smoke.py    29 tests, all runnable in mock mode
```

## How an event is formed
1. The camera thread always holds the newest frame, so inference never builds a backlog.
2. Main loop: preprocess (`clahe_gray` by default) → YOLO. A hit above 0.45 confidence opens a pending event.
3. After `FUSION_DELAY_S` (default 0.4 s): peak depth drop and peak Z-g are read from the 500 ms sensor window ending then, plus the latest GPS fix (synthetic Mumbai route if there's no fix).
4. Dedup drops the event if it's < 3 s after the last one or < 5 m from the last location. Otherwise severity is set (LOW / MEDIUM / CRITICAL), and the event is written to `output/events.jsonl` and optionally POSTed to `TELEMETRY_URL`.
5. Depth or IMU spikes with no vision hit also emit events (`vision.detected: false`); set `ENABLE_SENSOR_ONLY_TRIGGERS=0` to turn this off.

Sensors that aren't connected disable themselves in `auto` mode (IMU → `z_impact_g: null`, ultrasonic → `depth_cm: null`), so the daemon runs with whatever hardware is present.

## Payload
```json
{
  "node_id": "CIVICOPS-PI4-01",
  "timestamp": "2026-09-25T10:43:48.136+00:00",
  "gps": {"latitude": 19.2812542, "longitude": 72.8561909, "is_mock": true},
  "vision": {"detected": true, "confidence": 0.875, "bbox": [184, 0, 303, 51], "area_px": 6069},
  "kinetics": {"z_impact_g": 1.485, "baseline_g": 1.0},
  "surface": {"depth_cm": 4.77, "baseline_distance_cm": 30.0},
  "severity_level": "MEDIUM"
}
```
Severity is **CRITICAL** if depth ≥ 8 cm or impact ≥ 2.5 g, and **MEDIUM** if depth ≥ 4 cm, impact ≥ 1.8 g, or the bbox covers ≥ 8 % of the frame. Everything else is **LOW**. All thresholds are in `config.py`.
