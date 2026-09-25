"""
Hardware bring-up: test ONE sensor at a time on the Pi, with live readouts.

  python hw_check.py camera        # grabs frames, saves raw + preprocessed snapshots
  python hw_check.py ultrasonic    # prints distance 5x/sec (wave your hand under it)
  python hw_check.py imu           # prints Z g (tap the board)
  python hw_check.py gps           # prints raw NMEA + fix status
  python hw_check.py detector      # runs the model on one camera frame
  python hw_check.py all           # quick pass/fail of everything

Ctrl+C to stop any live readout.
"""
from __future__ import annotations

import logging
import sys
import time

import cv2

import config
from sensors.camera import Camera, preprocess
from sensors.gps_node import GPSNode
from sensors.imu import IMU
from sensors.ultrasonic import Ultrasonic

logging.basicConfig(level="INFO", format="%(levelname)-7s %(name)-10s %(message)s")


def check_camera(live: bool = True) -> bool:
    try:
        cam = Camera(mode="hardware" if not config.MOCK_HARDWARE else None).start()
    except RuntimeError as e:
        print(f"FAIL camera: {e}\n   Check: rpicam-hello --list-cameras  (ribbon orientation, power off when connecting)")
        return False
    try:
        frame = cam.read(timeout=5)
        if frame is None:
            print("FAIL camera: no frame")
            return False
        out = config.OUTPUT_DIR
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "check_raw.jpg"), frame)
        for mode in ("grayworld", "clahe_gray"):
            cv2.imwrite(str(out / f"check_{mode}.jpg"), preprocess(frame, mode))
        print(f"OK camera [{cam.backend}] {frame.shape} -> snapshots in {out}/check_*.jpg")
        if live:
            t0, n0 = time.time(), cam.frames_captured
            time.sleep(3)
            print(f"   capture rate: {(cam.frames_captured - n0) / (time.time() - t0):.1f} FPS")
        return True
    finally:
        cam.stop()


def check_ultrasonic(live: bool = True) -> bool:
    us = Ultrasonic().start()
    try:
        if not us.enabled:
            print("FAIL ultrasonic: disabled (gpiozero missing or wiring issue)")
            return False
        time.sleep(1)
        d = us.latest_distance()
        print(f"OK ultrasonic [{us.backend}] distance={d} cm baseline={us.baseline_cm} cm timeouts={us.timeouts}")
        while live:
            print(f"   {us.latest_distance()!s:>8} cm   depth(0.5s)={us.max_depth(0.5)}   timeouts={us.timeouts}")
            time.sleep(0.2)
        return d is not None
    except KeyboardInterrupt:
        return True
    finally:
        us.stop()


def check_imu(live: bool = True) -> bool:
    imu = IMU().start()
    try:
        if not imu.enabled:
            print("SKIP imu: not detected (fine - payload z_impact_g will be null). "
                  "Check with: i2cdetect -y 1  (expect 68)")
            return True
        time.sleep(1)
        print(f"OK imu [{imu.backend}] samples/s~{imu.samples} peakZ={imu.peak_z(0.5)} g")
        while live:
            print(f"   peak Z (0.5s) = {imu.peak_z(0.5)} g")
            time.sleep(0.2)
        return True
    except KeyboardInterrupt:
        return True
    finally:
        imu.stop()


def check_gps(live: bool = True) -> bool:
    gps = GPSNode().start()
    try:
        if config.MOCK_HARDWARE:
            p = gps.position()
            print(f"OK gps [synthetic/mock] {p.latitude:.5f},{p.longitude:.5f}")
            return True
        if gps.backend != "serial":
            print(f"WARN gps: serial {config.GPS_PORT} not available -> synthetic coordinates")
            return False
        time.sleep(3)
        print(f"OK gps [serial] sentences={gps.sentences} bad_checksums={gps.bad_checksums} fix={gps.has_fix()}")
        if gps.sentences == 0:
            print("   No NMEA at all: check TX->GPIO15(pin10), baud 9600, serial enabled, console disabled")
        while live:
            p = gps.position()
            print(f"   fix={gps.has_fix()}  {p.latitude:.6f},{p.longitude:.6f}  mock={p.is_mock}  "
                  f"sats={p.satellites}  sentences={gps.sentences}")
            time.sleep(1)
        return True
    except KeyboardInterrupt:
        return True
    finally:
        gps.stop()


def check_detector() -> bool:
    from model.detector import Detector
    det = Detector().load()
    cam = Camera().start()
    try:
        frame = cam.read(timeout=5)
        dets = det.detect(preprocess(frame))
        dets = det.detect(preprocess(frame))   # second call = warm timing
        print(f"OK detector [{det.backend}] {det.last_inference_ms:.0f} ms, detections: "
              f"{[(d.class_name, round(d.confidence, 2)) for d in dets]}")
        return True
    finally:
        cam.stop()


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    checks = {"camera": check_camera, "ultrasonic": check_ultrasonic, "imu": check_imu, "gps": check_gps}
    if what == "detector":
        check_detector()
    elif what == "all":
        results = {name: fn(live=False) for name, fn in checks.items()}
        print("\nSUMMARY:", "  ".join(f"{k}={'OK' if v else 'CHECK'}" for k, v in results.items()))
    elif what in checks:
        checks[what](live=True)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
