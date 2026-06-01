# Braille Dot Detection Model Training Guide (Phase E)

This guide documents the procedures for dataset preparation, YOLOv8 model training recommendations, and validation testing for Phase E of the **BrailleVisionAI** system.

---

## 1. Dataset Preparation Guide

To train a robust YOLOv8 dot detector, you need labeled bounding box annotations locating individual Braille dots in various images.

### Directory Structure

```
datasets/braille_dots/
├── images/
│   ├── train/
│   │   ├── dot_img_001.jpg
│   │   └── ...
│   └── val/
│       ├── dot_img_100.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── dot_img_001.txt
    │   └── ...
    └── val/
        ├── dot_img_100.txt
        └── ...
```

### Label Format (YOLO v8)

Each image requires a matching `.txt` file containing bounding box coordinates normalized to `[0, 1]`.
Format per dot:
```
<class_id> <x_center> <y_center> <width> <height>
```
* Since there is only one class (`dot`), `<class_id>` is always `0`.
* Example line in `labels/train/dot_img_001.txt`:
  `0 0.452 0.312 0.024 0.024` (specifies a dot centered at 45.2% width and 31.2% height, with a diameter of 2.4% of the image size).

---

## 2. YOLOv8 Training Recommendations

YOLOv8-nano (`yolov8n.pt`) is recommended due to its small footprint and extreme fast inference speed (~2-5ms on standard CPUs), making it ideal for live webcam streams.

### Configuration file (`dataset.yaml`)

Create a `dataset.yaml` config file:
```yaml
path: ../datasets/braille_dots  # dataset root dir (relative to training run or absolute)
train: images/train
val: images/val

names:
  0: dot
```

### Training Command

Run the training pipeline via the Ultralytics CLI:
```bash
yolo detect train \
  data=datasets/braille_dots/dataset.yaml \
  model=yolov8n.pt \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  project=models/braille_dots \
  name=braille_dots_yolov8
```

### Post-Training Integration

Once training is complete:
1. Locate the trained weights: `models/braille_dots/braille_dots_yolov8/weights/best.pt`.
2. Copy and rename this file to: `models/braille_dots/braille_dots_yolov8.pt`.
3. The system will **automatically detect** the file on startup and run YOLO-based dot segmentation instead of the OpenCV blob thresholding fallback.

---

## 3. Testing Instructions

### Mock/Local Testing
1. Upload a clear Braille image in the **Upload & Analyze** page of the app.
2. Verify that:
   - Green circles are rendered around individual dots.
   - Dotted cyan/yellow boxes frame grouped Braille cells.
   - Cell indicators (e.g. `#1 [A]`) appear on the banner above each cell.
   - Binary string patterns (e.g. `100000`) are drawn below each cell.
   - The side-panel displays a tabular layout of all extracted cells and their characters in reading order.
