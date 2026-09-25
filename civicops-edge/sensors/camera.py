"""
Camera acquisition: Picamera2 (Pi Camera v2 NoIR) -> OpenCV webcam -> mock.

A background thread grabs frames continuously and keeps only the newest one,
so slow YOLO inference never stalls capture (the inference loop always gets
the freshest frame instead of a growing backlog).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import config
from sensors.mock_road import get_road

log = logging.getLogger("camera")


# ---------------------------------------------------------------------------
# Pre-processing (NoIR colour-cast handling)
# ---------------------------------------------------------------------------
_clahe = None


def _get_clahe():
    global _clahe
    if _clahe is None:
        _clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                                 tileGridSize=(config.CLAHE_TILE_GRID, config.CLAHE_TILE_GRID))
    return _clahe


def gray_world(frame: np.ndarray) -> np.ndarray:
    """Software white balance: scale each channel so their means match.
    Keeps colour but removes most of the NoIR magenta cast in daylight."""
    f = frame.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0) + 1e-6
    f *= means.mean() / means
    return np.clip(f, 0, 255).astype(np.uint8)


def clahe_gray(frame: np.ndarray) -> np.ndarray:
    """Grayscale + CLAHE, replicated to 3 channels for YOLO.
    Colour-independent, so the NoIR cast disappears entirely and day/IR-night
    frames look alike to the model."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = _get_clahe().apply(gray)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess(frame: np.ndarray, mode: str | None = None) -> np.ndarray:
    mode = (mode or config.PREPROCESS_MODE).lower()
    if mode == "clahe_gray":
        return clahe_gray(frame)
    if mode == "grayworld":
        return gray_world(frame)
    return frame


# ---------------------------------------------------------------------------
# Mock frame generator
# ---------------------------------------------------------------------------
class MockFrameSource:
    """Synthetic asphalt texture; a dark elliptical pothole rolls through the
    frame whenever the shared MockRoad says we are passing one."""

    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        rng = np.random.default_rng(config.MOCK_SEED)
        noise = rng.normal(120, 18, (height * 2, width)).clip(0, 255).astype(np.uint8)
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        tex = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        # NoIR-style magenta tint so the preprocessing path is exercised
        tex = cv2.add(tex, np.full_like(tex, (25, 0, 30)))
        self.texture = tex
        self.scroll = 0

    def read(self) -> np.ndarray:
        self.scroll = (self.scroll + 6) % self.h
        frame = self.texture[self.scroll:self.scroll + self.h].copy()
        # lane marking
        cv2.line(frame, (self.w // 2, 0), (self.w // 2, self.h), (200, 200, 210), 4)
        st = get_road().state()
        if st.active:
            cy = int(st.progress * self.h * 1.1)         # moves top -> bottom
            cx = int(self.w * 0.38)
            ax = int(self.w * st.width_frac / 2)
            ay = max(8, int(ax * 0.55))
            cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (28, 22, 30), -1)
            cv2.ellipse(frame, (cx, cy), (ax, ay), 0, 0, 360, (70, 65, 75), 3)
        return frame


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
class Camera:
    def __init__(self, mode: str | None = None) -> None:
        self.requested_mode = config.resolve_mode(mode or config.CAMERA_MODE)
        self.backend: str = "none"
        self.width, self.height = config.CAMERA_WIDTH, config.CAMERA_HEIGHT
        self._picam = None
        self._cap = None
        self._mock: Optional[MockFrameSource] = None
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._frame_ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_captured = 0

    # -- backend setup ------------------------------------------------------
    def _open_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2  # type: ignore
        except Exception as e:  # ImportError on non-Pi
            log.info("picamera2 unavailable (%s)", e)
            return False
        try:
            tuning = None
            if config.CAMERA_NOIR_TUNING:
                try:
                    tuning = Picamera2.load_tuning_file(config.CAMERA_NOIR_TUNING_FILE)
                    log.info("Loaded NoIR tuning file %s", config.CAMERA_NOIR_TUNING_FILE)
                except Exception as e:
                    log.warning("Could not load NoIR tuning (%s); using default", e)
            cam = Picamera2(tuning=tuning) if tuning else Picamera2()
            cfg = cam.create_video_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},  # BGR order in numpy -> OpenCV-ready
                controls={"FrameRate": config.CAMERA_FPS},
                buffer_count=4,
            )
            cam.configure(cfg)
            cam.start()
            time.sleep(0.5)  # let AE/AWB settle
            self._picam = cam
            self.backend = "picamera2"
            return True
        except Exception as e:
            log.warning("picamera2 failed to start: %s", e)
            return False

    def _open_opencv(self) -> bool:
        cap = cv2.VideoCapture(config.CAMERA_OPENCV_INDEX)
        if not cap or not cap.isOpened():
            log.info("OpenCV camera %s unavailable", config.CAMERA_OPENCV_INDEX)
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, config.CAMERA_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self.backend = "opencv"
        return True

    def _open_mock(self) -> bool:
        self._mock = MockFrameSource(self.width, self.height)
        self.backend = "mock"
        return True

    def open(self) -> None:
        mode = self.requested_mode
        if mode == "disabled":
            raise RuntimeError("Camera cannot be disabled - it is the primary trigger")
        if mode == "mock":
            self._open_mock()
        elif self._open_picamera2() or self._open_opencv():
            pass
        elif mode == "auto" and config.CAMERA_FALLBACK_TO_MOCK:
            log.warning("No camera found - falling back to MOCK frames")
            self._open_mock()
        else:
            raise RuntimeError("No camera available (picamera2 and /dev/video0 both failed)")
        log.info("Camera backend: %s (%dx%d)", self.backend, self.width, self.height)

    # -- raw grab -----------------------------------------------------------
    def _grab(self) -> Optional[np.ndarray]:
        if self.backend == "picamera2":
            return self._picam.capture_array("main")
        if self.backend == "opencv":
            ok, frame = self._cap.read()
            return frame if ok else None
        if self.backend == "mock":
            time.sleep(1.0 / config.CAMERA_FPS)
            return self._mock.read()
        return None

    def _loop(self) -> None:
        fails = 0
        while not self._stop.is_set():
            try:
                frame = self._grab()
            except Exception as e:
                log.error("Frame grab error: %s", e)
                frame = None
            if frame is None:
                fails += 1
                if fails % 50 == 1:
                    log.warning("Camera returned no frame (%d consecutive)", fails)
                time.sleep(0.02)
                continue
            fails = 0
            if frame.ndim == 3 and frame.shape[2] == 4:   # XBGR8888 -> BGR
                frame = frame[:, :, :3]
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_ts = time.monotonic()
            self.frames_captured += 1

    # -- public API -----------------------------------------------------------
    def start(self) -> "Camera":
        if self.backend == "none":
            self.open()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()
        return self

    def latest(self) -> Tuple[int, Optional[np.ndarray], float]:
        """Return (frame_id, frame_copy, monotonic_ts). frame is None until the first grab."""
        with self._lock:
            if self._frame is None:
                return 0, None, 0.0
            return self._frame_id, self._frame.copy(), self._frame_ts

    def read(self, timeout: float = 2.0) -> Optional[np.ndarray]:
        """Blocking convenience read (used by tests / one-shot scripts)."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            fid, frame, _ = self.latest()
            if frame is not None:
                return frame
            time.sleep(0.01)
        return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            if self._picam is not None:
                self._picam.stop()
                self._picam.close()
        except Exception as e:
            log.warning("picamera2 close error: %s", e)
        if self._cap is not None:
            self._cap.release()
        self._picam = self._cap = None
        log.info("Camera released")
