"""
YOLOv8n pothole detector.

Load order (first that exists): NCNN folder -> ONNX -> .pt  (see config.MODEL_CANDIDATES)
NCNN is the fastest option on the Pi 4's Cortex-A72 CPU.

NOTE: stock yolov8n is trained on COCO and has NO 'pothole' class. Use
pothole-trained weights (see download_assets.py --weights). Until then,
ALLOW_ANY_CLASS_FALLBACK lets any COCO object trigger so the pipeline can be tested.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

import config

log = logging.getLogger("detector")


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: List[int]          # [x1, y1, x2, y2] in pixels

    @property
    def area_px(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)


class Detector:
    def __init__(self, mode: str | None = None, model_path: str | None = None) -> None:
        self.requested_mode = config.resolve_mode(mode or config.DETECTOR_MODE)
        self.model_path = model_path
        self.backend = "none"
        self.model = None
        self.names: dict = {}
        self.target_ids: Optional[set] = None   # None = accept any class
        self.last_inference_ms = 0.0

    # ------------------------------------------------------------------
    def load(self) -> "Detector":
        if self.requested_mode == "mock":
            self.backend = "mock"
            self.names = {0: "pothole"}
            self.target_ids = {0}
            log.info("Detector backend: mock (dark-blob finder)")
            return self

        candidates = [self.model_path] if self.model_path else config.MODEL_CANDIDATES
        path = next((p for p in candidates if p and Path(p).exists()), None)
        if path is None:
            raise FileNotFoundError(
                "No model found. Run:  python -m model.download_assets --export\n"
                f"Looked for: {candidates}")
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as e:
            raise RuntimeError("ultralytics not installed: pip install ultralytics") from e

        self.model = YOLO(path)   # task (detect/segment) read from model metadata
        self.backend = ("ncnn" if Path(path).is_dir() else Path(path).suffix.lstrip("."))
        self.names = dict(self.model.names) if getattr(self.model, "names", None) else {}
        wanted = {n.lower() for n in config.TARGET_CLASSES}
        ids = {i for i, n in self.names.items() if str(n).lower() in wanted}
        if ids:
            self.target_ids = ids
        elif config.ALLOW_ANY_CLASS_FALLBACK:
            self.target_ids = None
            log.warning("Model %s has none of %s - accepting ANY class (test mode). "
                        "Load pothole weights for real use.", path, config.TARGET_CLASSES)
        else:
            raise RuntimeError(f"Model has no classes named {config.TARGET_CLASSES}: {self.names}")

        # warm-up (first NCNN/ONNX call is slow)
        dummy = np.zeros((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), dtype=np.uint8)
        self._predict(dummy)
        log.info("Detector backend: %s (%s), classes=%s", self.backend, path,
                 "ANY" if self.target_ids is None else [self.names[i] for i in self.target_ids])
        return self

    # ------------------------------------------------------------------
    def _predict(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.predict(frame, imgsz=config.INFERENCE_IMGSZ,
                                 conf=config.DETECTION_CONFIDENCE, verbose=False)[0]
        out: List[Detection] = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        clss = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls)
        for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
            k = int(k)
            if self.target_ids is not None and k not in self.target_ids:
                continue
            out.append(Detection(str(self.names.get(k, k)), float(c),
                                 [int(x1), int(y1), int(x2), int(y2)]))
        return out

    def _predict_mock(self, frame: np.ndarray) -> List[Detection]:
        """Find the synthetic pothole: a large, very dark blob."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        thr = max(10.0, float(np.median(gray)) * 0.45)
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = gray.shape
        out: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.004 * w * h:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            fill = area / float(bw * bh)                       # ellipse ~0.78
            conf = float(np.clip(0.35 + 0.6 * fill + 2.0 * area / (w * h), 0.0, 0.97))
            if conf >= config.DETECTION_CONFIDENCE:
                out.append(Detection("pothole", round(conf, 3), [x, y, x + bw, y + bh]))
        return out

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.backend == "none":
            self.load()
        t = time.perf_counter()
        dets = self._predict_mock(frame) if self.backend == "mock" else self._predict(frame)
        self.last_inference_ms = (time.perf_counter() - t) * 1000
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets
