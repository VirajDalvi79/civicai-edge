"""
CivicOps Edge — central configuration.

Every value can be overridden with an environment variable of the same name,
e.g.  MOCK_HARDWARE=1 python main.py
"""
from __future__ import annotations

import os
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val not in (None, "") else default


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val not in (None, "") else default


BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Global
# ---------------------------------------------------------------------------
NODE_ID = _env_str("NODE_ID", "CIVICOPS-PI4-01")

# Master switch: when True every sensor + the detector are simulated.
MOCK_HARDWARE = _env_bool("MOCK_HARDWARE", False)

# Per-sensor mode: "auto" | "hardware" | "mock" | "disabled"
#   auto     -> mock if MOCK_HARDWARE, else try hardware and degrade gracefully
#   hardware -> hardware only; raise if unavailable
#   mock     -> always simulate
#   disabled -> sensor off (fields become null in the payload)
CAMERA_MODE = _env_str("CAMERA_MODE", "auto")
ULTRASONIC_MODE = _env_str("ULTRASONIC_MODE", "auto")
IMU_MODE = _env_str("IMU_MODE", "auto")          # no MPU-6050 yet -> auto disables itself
GPS_MODE = _env_str("GPS_MODE", "auto")
DETECTOR_MODE = _env_str("DETECTOR_MODE", "auto")

LOG_LEVEL = _env_str("LOG_LEVEL", "INFO")
STATS_INTERVAL_S = _env_float("STATS_INTERVAL_S", 10.0)

# ---------------------------------------------------------------------------
# Camera (Pi Camera v2 NoIR, IMX219, CSI)
# ---------------------------------------------------------------------------
CAMERA_WIDTH = _env_int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _env_int("CAMERA_HEIGHT", 480)
CAMERA_FPS = _env_int("CAMERA_FPS", 30)
# Load libcamera's NoIR-specific tuning (fixes most of the pink/purple cast).
CAMERA_NOIR_TUNING = _env_bool("CAMERA_NOIR_TUNING", True)
CAMERA_NOIR_TUNING_FILE = _env_str("CAMERA_NOIR_TUNING_FILE", "imx219_noir.json")
# Fallback USB webcam index (/dev/video0)
CAMERA_OPENCV_INDEX = _env_int("CAMERA_OPENCV_INDEX", 0)
# If no real camera works, fall back to synthetic frames instead of exiting.
CAMERA_FALLBACK_TO_MOCK = _env_bool("CAMERA_FALLBACK_TO_MOCK", False)
# Pre-processing before YOLO: "none" | "grayworld" | "clahe_gray"
PREPROCESS_MODE = _env_str("PREPROCESS_MODE", "clahe_gray")
CLAHE_CLIP_LIMIT = _env_float("CLAHE_CLIP_LIMIT", 2.0)
CLAHE_TILE_GRID = _env_int("CLAHE_TILE_GRID", 8)

# ---------------------------------------------------------------------------
# Detector (YOLOv8n)
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "model"
# Tried in order; first that exists is loaded.
MODEL_CANDIDATES = [
    _env_str("MODEL_PATH", str(MODEL_DIR / "pothole_ncnn_model")),
    str(MODEL_DIR / "yolov8n_ncnn_model"),
    str(MODEL_DIR / "pothole.onnx"),
    str(MODEL_DIR / "yolov8n.onnx"),
    str(MODEL_DIR / "pothole.pt"),
    str(MODEL_DIR / "yolov8n.pt"),
]
DETECTION_CONFIDENCE = _env_float("DETECTION_CONFIDENCE", 0.45)
INFERENCE_IMGSZ = _env_int("INFERENCE_IMGSZ", 320)     # 320 is ~2-3x faster than 640 on a Pi 4
TARGET_CLASSES = [c.strip() for c in _env_str("TARGET_CLASSES", "pothole").split(",") if c.strip()]
# If the model has none of TARGET_CLASSES (e.g. stock COCO yolov8n), accept any class.
# Useful for pipeline testing; turn off once you have pothole weights.
ALLOW_ANY_CLASS_FALLBACK = _env_bool("ALLOW_ANY_CLASS_FALLBACK", True)
SAVE_EVENT_IMAGES = _env_bool("SAVE_EVENT_IMAGES", True)

# ---------------------------------------------------------------------------
# HC-SR04 ultrasonic (depth)
# ---------------------------------------------------------------------------
ULTRASONIC_TRIG_PIN = _env_int("ULTRASONIC_TRIG_PIN", 23)   # physical pin 16
ULTRASONIC_ECHO_PIN = _env_int("ULTRASONIC_ECHO_PIN", 24)   # physical pin 18 (via 1k/2k divider!)
ULTRASONIC_MAX_DISTANCE_CM = _env_float("ULTRASONIC_MAX_DISTANCE_CM", 400.0)
ULTRASONIC_MIN_DISTANCE_CM = _env_float("ULTRASONIC_MIN_DISTANCE_CM", 2.0)
ULTRASONIC_SAMPLE_HZ = _env_float("ULTRASONIC_SAMPLE_HZ", 20.0)
ULTRASONIC_BUFFER_S = _env_float("ULTRASONIC_BUFFER_S", 5.0)
# Flat-road distance (sensor -> road). Replaced by calibration file if present.
ULTRASONIC_DEFAULT_BASELINE_CM = _env_float("ULTRASONIC_DEFAULT_BASELINE_CM", 30.0)
ULTRASONIC_BASELINE_FILE = Path(_env_str("ULTRASONIC_BASELINE_FILE", str(BASE_DIR / "calibration" / "ultrasonic_baseline.json")))
ULTRASONIC_CALIBRATION_S = _env_float("ULTRASONIC_CALIBRATION_S", 5.0)
# Depth drops smaller than this are treated as road noise.
DEPTH_NOISE_FLOOR_CM = _env_float("DEPTH_NOISE_FLOOR_CM", 1.0)

# ---------------------------------------------------------------------------
# MPU-6050 IMU
# ---------------------------------------------------------------------------
IMU_I2C_BUS = _env_int("IMU_I2C_BUS", 1)
IMU_I2C_ADDRESS = int(_env_str("IMU_I2C_ADDRESS", "0x68"), 16)
IMU_SAMPLE_HZ = _env_float("IMU_SAMPLE_HZ", 100.0)
IMU_ACCEL_RANGE_G = _env_int("IMU_ACCEL_RANGE_G", 8)        # 2 | 4 | 8 | 16
IMU_BUFFER_S = _env_float("IMU_BUFFER_S", 5.0)
IMU_IMPACT_THRESHOLD_G = _env_float("IMU_IMPACT_THRESHOLD_G", 1.5)
BASELINE_G = 1.0

# ---------------------------------------------------------------------------
# GPS (u-blox NEO-6M on UART)
# ---------------------------------------------------------------------------
GPS_PORT = _env_str("GPS_PORT", "/dev/serial0")
GPS_BAUD = _env_int("GPS_BAUD", 9600)
GPS_FIX_TIMEOUT_S = _env_float("GPS_FIX_TIMEOUT_S", 5.0)    # no fix for this long -> synthetic
GPS_MOCK_SPEED_MPS = _env_float("GPS_MOCK_SPEED_MPS", 8.0)   # ~29 km/h
# Mira Road -> Kashimira -> Western Express Highway (south towards Dahisar/Borivali)
GPS_MOCK_ROUTE = [
    (19.2812, 72.8561),
    (19.2849, 72.8623),
    (19.2890, 72.8680),
    (19.2905, 72.8697),   # Kashimira junction
    (19.2800, 72.8688),
    (19.2690, 72.8676),
    (19.2595, 72.8665),   # Dahisar check naka
    (19.2480, 72.8650),
    (19.2350, 72.8640),
]

# ---------------------------------------------------------------------------
# Fusion / events
# ---------------------------------------------------------------------------
FUSION_WINDOW_S = _env_float("FUSION_WINDOW_S", 0.5)
# Delay before sampling depth/IMU after a vision hit. The camera looks AHEAD of
# the ultrasonic/IMU, so at speed the wheel reaches the pothole a bit later.
# The sensor window is [t_detect + DELAY - WINDOW, t_detect + DELAY].
# 0.0 = "last 500 ms at detection time" (literal spec). 0.4 also covers the
# moment the wheel/sensor actually crosses the pothole at city speeds.
FUSION_DELAY_S = _env_float("FUSION_DELAY_S", 0.4)
# Also emit events from depth/IMU alone (vision.detected = false)
ENABLE_SENSOR_ONLY_TRIGGERS = _env_bool("ENABLE_SENSOR_ONLY_TRIGGERS", True)
SENSOR_ONLY_DEPTH_CM = _env_float("SENSOR_ONLY_DEPTH_CM", 5.0)

# Severity rules (any rule met -> that level)
SEVERITY_CRITICAL_DEPTH_CM = _env_float("SEVERITY_CRITICAL_DEPTH_CM", 8.0)
SEVERITY_CRITICAL_G = _env_float("SEVERITY_CRITICAL_G", 2.5)
SEVERITY_MEDIUM_DEPTH_CM = _env_float("SEVERITY_MEDIUM_DEPTH_CM", 4.0)
SEVERITY_MEDIUM_G = _env_float("SEVERITY_MEDIUM_G", 1.8)
SEVERITY_MEDIUM_AREA_RATIO = _env_float("SEVERITY_MEDIUM_AREA_RATIO", 0.08)

# Deduplication
DEDUP_COOLDOWN_S = _env_float("DEDUP_COOLDOWN_S", 3.0)
DEDUP_RADIUS_M = _env_float("DEDUP_RADIUS_M", 5.0)
# Forget the last location after this long (so a revisit tomorrow still counts)
DEDUP_LOCATION_TTL_S = _env_float("DEDUP_LOCATION_TTL_S", 600.0)

# ---------------------------------------------------------------------------
# Telemetry dispatch
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(_env_str("OUTPUT_DIR", str(BASE_DIR / "output")))
TELEMETRY_JSONL = OUTPUT_DIR / "events.jsonl"
EVENT_IMAGE_DIR = OUTPUT_DIR / "event_images"
TELEMETRY_URL = _env_str("TELEMETRY_URL", "")     # empty = local file only
TELEMETRY_API_KEY = _env_str("TELEMETRY_API_KEY", "")
TELEMETRY_TIMEOUT_S = _env_float("TELEMETRY_TIMEOUT_S", 3.0)
TELEMETRY_MAX_RETRIES = _env_int("TELEMETRY_MAX_RETRIES", 3)
TELEMETRY_QUEUE_SIZE = _env_int("TELEMETRY_QUEUE_SIZE", 500)

# ---------------------------------------------------------------------------
# Mock simulation
# ---------------------------------------------------------------------------
MOCK_POTHOLE_PERIOD_S = _env_float("MOCK_POTHOLE_PERIOD_S", 6.0)   # one pothole every N s
MOCK_POTHOLE_DURATION_S = _env_float("MOCK_POTHOLE_DURATION_S", 1.2)
MOCK_SEED = _env_int("MOCK_SEED", 42)


def resolve_mode(mode: str) -> str:
    """Turn 'auto' into 'mock' when MOCK_HARDWARE is on."""
    mode = (mode or "auto").lower()
    if mode == "auto" and MOCK_HARDWARE:
        return "mock"
    return mode
