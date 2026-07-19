"""Export official YOLOv8 segmentation checkpoints to ONNX for OpenCV DNN.

Run from the repository root inside the project virtual environment:

    .venv\\Scripts\\python.exe scripts\\export_yolov8_seg.py

Missing `yolov8<size>-seg.pt` checkpoints are downloaded by Ultralytics into
the checkpoint directory. Note that the plain `yolov8<size>.pt` checkpoints
are detection models and cannot be used for segmentation.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from ultralytics import YOLO

SIZES = ("n", "s", "m", "l", "x")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("YOLOv8"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/yolov8"))
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for size in args.sizes:
        checkpoint = args.checkpoint_dir / f"yolov8{size}-seg.pt"
        model = YOLO(str(checkpoint))
        exported = Path(
            model.export(
                format="onnx",
                dynamic=False,
                simplify=True,
                imgsz=640,
            )
        )
        destination = args.output_dir / exported.name
        shutil.move(exported, destination)
        print(f"{destination.name}: checkpoint SHA-256 {sha256(checkpoint)}")
        print(f"{destination.name}: ONNX SHA-256 {sha256(destination)}")
        print(f"Exported: {destination} ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
