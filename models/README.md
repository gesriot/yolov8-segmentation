# Models

Model binaries are intentionally ignored by Git. Local YOLOv8 segmentation
exports live under:

```text
models/yolov8/yolov8{n,s,m,l,x}-seg.onnx
```

## YOLOv8 segmentation models

- Source: official Ultralytics YOLOv8 segmentation checkpoints
  (`yolov8{n,s,m,l,x}-seg`). Note that the plain `yolov8n.pt` … `yolov8x.pt`
  checkpoints are **detection** models and do not work with this project.
- Export: ONNX, static input, simplified graph, `imgsz=640`.
- Reproduce: `.venv\Scripts\python scripts\export_yolov8_seg.py` (checkpoints
  are taken from — and downloaded into — `YOLOv8/`).
- Classes: COCO 80 (`shared/classes/coco80.txt`).
- License: Ultralytics AGPL-3.0 unless covered by a separate Enterprise
  License.

OpenCV 4.14.0 loads and executes these exports. The unconnected output layer
names are `output0` and `output1`.

Observed static output shapes (640×640 input), identical for every size:

- `output0`: `[1, 116, 8400]` – anchor-free detection rows in feature-major
  layout; after transposition each row is
  `[cx, cy, w, h, class_score_0 … class_score_79, mask_coeff_0 … mask_coeff_31]`.
  YOLOv8 has no built-in NMS, so the decoder applies `cv::dnn::NMSBoxes`
  after confidence filtering;
- `output1`: `[1, 32, 160, 160]` – mask prototypes.
