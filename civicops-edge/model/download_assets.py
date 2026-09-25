"""
Model asset helper.

  # stock COCO yolov8n -> NCNN (pipeline testing only; no pothole class)
  python -m model.download_assets --export

  # your pothole-trained weights -> model/pothole_ncnn_model/  (what main.py loads first)
  python -m model.download_assets --weights path/to/best.pt

  # RECOMMENDED: pre-trained YOLOv8n pothole model (Hugging Face, keremberke) -> NCNN
  python -m model.download_assets --pothole

  # download weights from a URL first, then export
  python -m model.download_assets --url https://.../best.pt

  # quick speed test of whatever model main.py would load
  python -m model.download_assets --benchmark

Run the export ON THE PI (or any aarch64 box): NCNN export is architecture-neutral,
but doing it on the Pi guarantees matching ultralytics/ncnn versions.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

MODEL_DIR = config.MODEL_DIR
POTHOLE_URL = "https://huggingface.co/keremberke/yolov8n-pothole-segmentation/resolve/main/best.pt"


def _export(pt_path: Path, out_name: str, imgsz: int) -> Path:
    from ultralytics import YOLO
    model = YOLO(str(pt_path))
    print(f"Classes in {pt_path.name}: {model.names}")
    exported = Path(model.export(format="ncnn", imgsz=imgsz))
    target = MODEL_DIR / out_name
    if exported.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported), str(target))
    print(f"NCNN model ready: {target}")
    return target


def export_stock(imgsz: int) -> Path:
    from ultralytics import YOLO
    cwd = os.getcwd()
    os.chdir(MODEL_DIR)            # ultralytics downloads yolov8n.pt into cwd
    try:
        YOLO("yolov8n.pt")
    finally:
        os.chdir(cwd)
    return _export(MODEL_DIR / "yolov8n.pt", "yolov8n_ncnn_model", imgsz)


def download(url: str, dest: Path) -> Path:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def benchmark(n: int) -> None:
    import numpy as np
    from model.detector import Detector
    det = Detector(mode="hardware").load()
    frame = (np.random.rand(config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3) * 255).astype("uint8")
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        det.detect(frame)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    print(f"{det.backend}: median {ts[len(ts)//2]:.1f} ms, p90 {ts[int(len(ts)*0.9)]:.1f} ms "
          f"(~{1000/ts[len(ts)//2]:.1f} FPS) at imgsz={config.INFERENCE_IMGSZ}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", action="store_true", help="download yolov8n.pt and export to NCNN")
    ap.add_argument("--weights", type=Path, help="custom .pt to export as pothole_ncnn_model")
    ap.add_argument("--pothole", action="store_true",
                    help="download keremberke/yolov8n-pothole-segmentation and export to NCNN")
    ap.add_argument("--url", help="download custom .pt weights from URL, then export")
    ap.add_argument("--imgsz", type=int, default=config.INFERENCE_IMGSZ)
    ap.add_argument("--benchmark", type=int, nargs="?", const=30, metavar="N")
    args = ap.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    did = False
    if args.pothole:
        args.url = POTHOLE_URL
    if args.url:
        args.weights = download(args.url, MODEL_DIR / "pothole.pt")
    if args.weights:
        _export(args.weights, "pothole_ncnn_model", args.imgsz)
        did = True
    if args.export:
        export_stock(args.imgsz)
        did = True
    if args.benchmark:
        benchmark(args.benchmark)
        did = True
    if not did:
        ap.print_help()


if __name__ == "__main__":
    main()
