"""
CivicOps Edge daemon - Phase 1.

Threads:
  camera      -> grabs frames continuously, keeps only the newest
  ultrasonic  -> 20 Hz distance samples into a ring buffer
  imu         -> 100 Hz Z-axis g samples into a ring buffer (if fitted)
  gps         -> parses NMEA from UART (synthetic fallback)
  dispatch    -> writes JSONL / POSTs telemetry
  main        -> preprocess -> YOLO -> fusion -> dedup -> enqueue

Usage:
  python main.py                   # real hardware (auto-degrades per sensor)
  python main.py --mock            # everything simulated (any laptop)
  python main.py --mock --duration 30
  python main.py --calibrate       # measure flat-road baseline for the HC-SR04
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

# --mock must be applied before config is imported anywhere
if "--mock" in sys.argv:
    os.environ["MOCK_HARDWARE"] = "1"

import cv2  # noqa: E402

import config  # noqa: E402
from core.deduplicator import Deduplicator  # noqa: E402
from core.dispatcher import TelemetryDispatcher  # noqa: E402
from core.schema import (GPSInfo, KineticsInfo, SurfaceInfo, TelemetryPayload,  # noqa: E402
                         VisionInfo, classify_severity, utc_now_iso)
from model.detector import Detection, Detector  # noqa: E402
from sensors.camera import Camera, preprocess  # noqa: E402
from sensors.gps_node import GPSNode  # noqa: E402
from sensors.imu import IMU  # noqa: E402
from sensors.ultrasonic import Ultrasonic  # noqa: E402

log = logging.getLogger("main")


@dataclass
class PendingEvent:
    t_detect: float
    due: float
    detection: Optional[Detection]
    frame: Optional["object"]


class EdgeDaemon:
    def __init__(self, duration: Optional[float] = None) -> None:
        self.duration = duration
        self.stop_event = threading.Event()
        self.camera = Camera()
        self.ultrasonic = Ultrasonic()
        self.imu = IMU()
        self.gps = GPSNode()
        self.detector = Detector()
        self.dedup = Deduplicator()
        self.dispatcher = TelemetryDispatcher()
        self.pending: Optional[PendingEvent] = None
        self.events: List[dict] = []
        self.frames_processed = 0
        self.detections = 0
        self._started = False
        self._shut_down = False

    # ------------------------------------------------------------------
    def start(self) -> "EdgeDaemon":
        log.info("Starting %s (MOCK_HARDWARE=%s)", config.NODE_ID, config.MOCK_HARDWARE)
        self.dispatcher.start()
        self.gps.start()
        self.imu.start()
        self.ultrasonic.start()
        self.detector.load()
        self.camera.start()
        self._started = True
        return self

    def install_signal_handlers(self) -> None:
        def _handler(signum, _frame):
            log.info("Signal %s received - shutting down", signal.Signals(signum).name)
            self.stop_event.set()
        try:
            signal.signal(signal.SIGINT, _handler)
            signal.signal(signal.SIGTERM, _handler)
        except ValueError:
            pass   # not in main thread (e.g. some test runners)

    # ------------------------------------------------------------------
    def _fuse(self, t_detect: float, det: Optional[Detection], frame) -> Optional[dict]:
        """Build a payload from vision + the sensor window ending at t_detect + FUSION_DELAY_S."""
        window_end = t_detect + config.FUSION_DELAY_S
        depth = self.ultrasonic.max_depth(config.FUSION_WINDOW_S, now=window_end)
        z_g = self.imu.peak_z(config.FUSION_WINDOW_S, now=window_end)
        fix = self.gps.position()

        if not self.dedup.check(fix.latitude, fix.longitude):
            log.debug("Duplicate suppressed: %s", self.dedup.last_reason)
            return None

        frame_area = config.CAMERA_WIDTH * config.CAMERA_HEIGHT
        if det is not None:
            vision = VisionInfo(True, round(det.confidence, 3), det.bbox, det.area_px)
        else:
            vision = VisionInfo(False)
        payload = TelemetryPayload(
            node_id=config.NODE_ID,
            timestamp=utc_now_iso(),
            gps=GPSInfo(fix.latitude, fix.longitude, fix.is_mock),
            vision=vision,
            kinetics=KineticsInfo(z_g),
            surface=SurfaceInfo(depth, self.ultrasonic.baseline_cm if self.ultrasonic.enabled else None),
            severity_level=classify_severity(depth, z_g, vision.area_px, frame_area),
        ).to_dict()

        self.dispatcher.submit(payload)
        self.events.append(payload)
        log.info("EVENT %s | vision=%s conf=%.2f | depth=%s cm | z=%s g | gps=%.5f,%.5f%s",
                 payload["severity_level"], vision.detected, vision.confidence, depth, z_g,
                 fix.latitude, fix.longitude, " (mock)" if fix.is_mock else "")
        if config.SAVE_EVENT_IMAGES and frame is not None:
            self._save_image(frame, det, payload)
        return payload

    def _save_image(self, frame, det: Optional[Detection], payload: dict) -> None:
        try:
            config.EVENT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            img = frame.copy()
            if det is not None:
                x1, y1, x2, y2 = det.bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"{det.class_name} {det.confidence:.2f} {payload['severity_level']}",
                            (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            name = payload["timestamp"].replace(":", "-").replace("+", "_") + ".jpg"
            cv2.imwrite(str(config.EVENT_IMAGE_DIR / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        except Exception as e:
            log.warning("Could not save event image: %s", e)

    def _sensor_only_trigger(self, now: float) -> bool:
        if not config.ENABLE_SENSOR_ONLY_TRIGGERS:
            return False
        z = self.imu.peak_z(0.2, now=now)
        d = self.ultrasonic.max_depth(0.2, now=now)
        return ((z is not None and z >= config.IMU_IMPACT_THRESHOLD_G) or
                (d is not None and d >= config.SENSOR_ONLY_DEPTH_CM))

    # ------------------------------------------------------------------
    def step(self) -> bool:
        """Process one new frame. Returns False if no new frame was available."""
        fid, frame, _ts = self.camera.latest()
        if frame is None or fid == getattr(self, "_last_fid", -1):
            now = time.monotonic()
            self._resolve_pending(now)
            return False
        self._last_fid = fid

        proc = preprocess(frame)
        dets = self.detector.detect(proc)
        now = time.monotonic()
        self.frames_processed += 1

        if dets:
            self.detections += 1
            best = dets[0]
            if self.pending is None:
                self.pending = PendingEvent(now, now + config.FUSION_DELAY_S, best, frame)
            elif best.confidence > (self.pending.detection.confidence if self.pending.detection else 0):
                self.pending.detection, self.pending.frame = best, frame   # keep the clearest view
        elif self.pending is None and self._sensor_only_trigger(now):
            self._fuse(now - config.FUSION_DELAY_S, None, frame)

        self._resolve_pending(now)
        return True

    def _resolve_pending(self, now: float) -> None:
        p = self.pending
        if p is not None and now >= p.due:
            self.pending = None
            self._fuse(p.t_detect, p.detection, p.frame)

    def run(self) -> None:
        t_start = last_stats = time.monotonic()
        frames_at_last = 0
        try:
            if not self._started:
                self.start()
            while not self.stop_event.is_set():
                if self.duration and time.monotonic() - t_start >= self.duration:
                    break
                if not self.step():
                    time.sleep(0.003)
                if time.monotonic() - last_stats >= config.STATS_INTERVAL_S:
                    dt = time.monotonic() - last_stats
                    log.info("stats: %.1f FPS processed | infer %.0f ms | cam %d fr | events %d | "
                             "suppressed %d | us %s | imu %s | gps %s",
                             (self.frames_processed - frames_at_last) / dt,
                             self.detector.last_inference_ms, self.camera.frames_captured,
                             len(self.events), self.dedup.suppressed, self.ultrasonic.backend,
                             self.imu.backend, "fix" if self.gps.has_fix() else "synthetic")
                    last_stats, frames_at_last = time.monotonic(), self.frames_processed
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shut_down:
            return
        self._shut_down = True
        log.info("Shutting down...")
        if self.pending is not None:                       # don't lose an in-flight event
            p, self.pending = self.pending, None
            try:
                self._fuse(p.t_detect, p.detection, p.frame)
            except Exception as e:
                log.warning("pending event lost: %s", e)
        for name, part in (("camera", self.camera), ("ultrasonic", self.ultrasonic),
                           ("imu", self.imu), ("gps", self.gps), ("dispatcher", self.dispatcher)):
            try:
                part.stop()
            except Exception as e:
                log.warning("%s stop error: %s", name, e)
        log.info("Shutdown complete: %d frames, %d events", self.frames_processed, len(self.events))


def main() -> None:
    ap = argparse.ArgumentParser(description="CivicOps edge sensing daemon")
    ap.add_argument("--mock", action="store_true", help="simulate all hardware")
    ap.add_argument("--duration", type=float, help="stop after N seconds")
    ap.add_argument("--calibrate", action="store_true", help="calibrate HC-SR04 flat-road baseline")
    ap.add_argument("--log-level", default=config.LOG_LEVEL)
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)-7s %(name)-10s %(message)s",
                        datefmt="%H:%M:%S")

    if args.calibrate:
        us = Ultrasonic()
        us.open()
        if not us.enabled:
            sys.exit("Ultrasonic sensor not available - check wiring (and the voltage divider!)")
        try:
            print(f"Baseline: {us.calibrate():.2f} cm  -> saved to {config.ULTRASONIC_BASELINE_FILE}")
        finally:
            us.stop()
        return

    daemon = EdgeDaemon(duration=args.duration)
    daemon.install_signal_handlers()
    try:
        daemon.run()
    except (RuntimeError, FileNotFoundError) as e:
        log.error("Startup failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
