"""Telemetry payload schema (plain dataclasses - no pydantic needed on the Pi)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import config

SEVERITY_LEVELS = ("LOW", "MEDIUM", "CRITICAL")


@dataclass
class GPSInfo:
    latitude: float
    longitude: float
    is_mock: bool


@dataclass
class VisionInfo:
    detected: bool
    confidence: float = 0.0
    bbox: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    area_px: int = 0


@dataclass
class KineticsInfo:
    z_impact_g: Optional[float]           # None when no IMU is fitted
    baseline_g: float = config.BASELINE_G


@dataclass
class SurfaceInfo:
    depth_cm: Optional[float]             # None when ultrasonic is off / no samples
    baseline_distance_cm: Optional[float]


@dataclass
class TelemetryPayload:
    node_id: str
    timestamp: str
    gps: GPSInfo
    vision: VisionInfo
    kinetics: KineticsInfo
    surface: SurfaceInfo
    severity_level: str

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), **kw)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def classify_severity(depth_cm: Optional[float], z_g: Optional[float],
                      area_px: int = 0, frame_area_px: int = 0) -> str:
    """Rule-based severity. Any matching rule sets the level; missing sensors are ignored."""
    d = depth_cm or 0.0
    g = z_g or 0.0
    ratio = (area_px / frame_area_px) if frame_area_px else 0.0
    if d >= config.SEVERITY_CRITICAL_DEPTH_CM or g >= config.SEVERITY_CRITICAL_G:
        return "CRITICAL"
    if (d >= config.SEVERITY_MEDIUM_DEPTH_CM or g >= config.SEVERITY_MEDIUM_G
            or ratio >= config.SEVERITY_MEDIUM_AREA_RATIO):
        return "MEDIUM"
    return "LOW"


def validate_payload(data: dict) -> None:
    """Raise ValueError if the dict doesn't match the telemetry contract."""
    req = {"node_id": str, "timestamp": str, "gps": dict, "vision": dict,
           "kinetics": dict, "surface": dict, "severity_level": str}
    for k, t in req.items():
        if k not in data or not isinstance(data[k], t):
            raise ValueError(f"missing/invalid field: {k}")
    datetime.fromisoformat(data["timestamp"])
    g = data["gps"]
    if not (-90 <= g["latitude"] <= 90 and -180 <= g["longitude"] <= 180):
        raise ValueError("gps out of range")
    if not isinstance(g["is_mock"], bool):
        raise ValueError("gps.is_mock must be bool")
    v = data["vision"]
    if not isinstance(v["detected"], bool) or len(v["bbox"]) != 4:
        raise ValueError("bad vision block")
    if not 0.0 <= v["confidence"] <= 1.0:
        raise ValueError("confidence out of range")
    if "z_impact_g" not in data["kinetics"] or data["kinetics"].get("baseline_g") != 1.0:
        raise ValueError("bad kinetics block")
    if "depth_cm" not in data["surface"] or "baseline_distance_cm" not in data["surface"]:
        raise ValueError("bad surface block")
    if data["severity_level"] not in SEVERITY_LEVELS:
        raise ValueError("bad severity_level")
