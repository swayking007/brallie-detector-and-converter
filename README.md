# 🦾 BrailleVisionAI

> **An AI-powered real-time accessibility assistant that reads embossed/handwritten Braille using a camera and converts it into English text and speech.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=flat-square&logo=opencv)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Phase](https://img.shields.io/badge/Phase-C%20Complete-22c55e?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 Problem Statement

Millions of visually impaired people worldwide rely on Braille as their primary written language. However:

- Braille readers are expensive and not universally accessible.
- Sighted caregivers, teachers, and family members often cannot read Braille.
- There is no affordable, portable, real-time solution to bridge this communication gap.

**BrailleVisionAI** aims to solve this by turning any camera (webcam, smartphone) into a live Braille-to-English translator powered by AI.

---

## 🎯 Vision & Objective

Build an end-to-end AI pipeline that:

1. **Captures** live camera input or static images containing embossed/handwritten Braille.
2. **Analyzes** image quality in real time and guides the user to a perfect capture.
3. **Detects** individual Braille dot patterns using a computer vision model.
4. **Translates** dot patterns into Grade-1 English Braille characters.
5. **Speaks** the translated text aloud using text-to-speech.
6. **Displays** results in a clean, accessible Streamlit web interface.

---

## 🏗️ Architecture Overview

```
Camera / Image Input
        │
        ▼
┌─────────────────────────┐
│  Phase C: Quality AI    │  ← Blur · Brightness · Alignment · Visibility
│  Smart Image Analyzer   │  ← Real-time guidance & corrections
└─────────────────────────┘
        │  (only if quality_ok)
        ▼
┌─────────────────────────┐
│  Phase D: Preprocessing │  ← Grayscale, denoise, threshold, edge detect
│  + Braille Detection    │  ← YOLOv8 / CNN to locate dot cells
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Phase E: Translation   │  ← Map dot positions → Braille → English
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Phase F: Voice Output  │  ← pyttsx3 / gTTS text-to-speech
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Streamlit UI           │  ← Live feed, quality panel, guidance, output
└─────────────────────────┘
```

---

## ✅ Phase Status

| Phase | Feature | Status |
|:---:|---|:---:|
| A | Project Foundation & Architecture | ✅ Complete |
| B | Camera System & Image Input (Webcam + Upload) | ✅ Complete |
| **C** | **Smart Quality Analyzer & AI Guidance System** | ✅ **Complete** |
| D | Image Preprocessing + YOLOv8 Braille Detection | 🔜 Next |
| E | Braille Translation Engine | 🔜 Planned |
| F | Text-to-Speech Voice Output | 🔜 Planned |

---

## 🔬 Phase C — Smart Image Quality Analyzer

### What Phase C adds

Phase C introduces a complete real-time image quality analysis system that evaluates every captured frame across 4 dimensions:

| Check | Method | Output |
|---|---|---|
| **Blur Detection** | OpenCV Laplacian variance | Clear / Slightly Blurry / Very Blurry |
| **Brightness Analysis** | Mean grayscale intensity + std-dev contrast | Too Dark / Good Lighting / Overexposed |
| **Alignment / Tilt** | Probabilistic Hough line transform (median angle) | Properly Aligned / Tilted Left / Tilted Right |
| **Distance / Visibility** | Canny edge pixel density ratio | Good Distance / Move Closer / Move Further |

### Phase C File Structure

```
preprocessing/
├── __init__.py
├── blur_detector.py        ← Laplacian variance sharpness score
├── brightness_analyzer.py  ← Mean intensity + contrast check
├── alignment_checker.py    ← Hough lines → tilt angle
├── visibility_estimator.py ← Canny edge density → distance proxy
└── quality_analyzer.py     ← Orchestrator — runs all 4 analyzers

ui/
├── __init__.py
├── guidance_panel.py       ← Full hackathon-ready Streamlit panel
└── camera_ui.py            ← Phase B camera management (unchanged)

app.py                      ← Streamlit entry point (Phase B + C)
```

### Blur Detector (`blur_detector.py`)

Uses **OpenCV Laplacian variance** — the gold standard method for blur detection:
1. Convert frame to grayscale
2. Apply Laplacian filter (highlights edges and fine detail)
3. Compute variance → higher = sharper

```python
from preprocessing.blur_detector import detect_blur
result = detect_blur(frame)
# result.score   → float (Laplacian variance, e.g., 245.3)
# result.status  → BlurStatus.CLEAR | SLIGHT | VERY_BLURRY
# result.is_ok   → True/False
# result.tip     → "✅ Sharp image — excellent for Braille detection."
# result.pct     → 0-100 for progress bar
```

**Recommended thresholds:**

| Threshold | Value | Meaning |
|---|---|---|
| `BLUR_CLEAR_THRESHOLD` | 120.0 | Above → Clear Image |
| `BLUR_SLIGHT_THRESHOLD` | 60.0 | 60–120 → Slightly Blurry |

### Brightness Analyzer (`brightness_analyzer.py`)

Dual-metric analysis — mean intensity AND contrast (std-dev):

```python
from preprocessing.brightness_analyzer import analyze_brightness
result = analyze_brightness(frame)
# result.score        → float mean pixel 0-255 (e.g., 142.6)
# result.std_dev      → float contrast measure (e.g., 38.2)
# result.status       → BrightnessStatus.DARK | GOOD | OVEREXPOSED
# result.low_contrast → True if std_dev < 20 (flat/washed out image)
# result.is_ok        → True/False
```

**Recommended thresholds:**

| Threshold | Value | Meaning |
|---|---|---|
| `BRIGHT_DARK_THRESHOLD` | 60 | Below → Too Dark |
| `BRIGHT_OVER_THRESHOLD` | 210 | Above → Overexposed |
| `CONTRAST_MIN` | 20 | Std-dev below → Low Contrast |

### Alignment Checker (`alignment_checker.py`)

Uses **Probabilistic Hough Line Transform** to detect dominant page tilt:
1. Grayscale → Gaussian blur → Canny edges
2. HoughLinesP finds line segments
3. Compute angle of each segment, take median → dominant tilt

```python
from preprocessing.alignment_checker import check_alignment
result = check_alignment(frame)
# result.angle      → float degrees (e.g., -3.5 = tilted left 3.5°)
# result.status     → AlignmentStatus.ALIGNED | TILTED_LEFT | TILTED_RIGHT
# result.line_count → int debug info
# result.pct        → 0-100 alignment quality score
```

**Recommended threshold:**

| Threshold | Value | Meaning |
|---|---|---|
| `TILT_THRESHOLD` | 8.0° | Within ±8° = Properly Aligned |

### Visibility Estimator (`visibility_estimator.py`)

Uses **Canny edge density** as a lightweight distance proxy:
- Too close → massive edge density (texture fills frame)
- Too far → very low edge density (page appears small)
- Good → medium density (dots visible, page fills frame)

```python
from preprocessing.visibility_estimator import estimate_visibility
result = estimate_visibility(frame)
# result.edge_density → float ratio 0.0-1.0 (e.g., 0.087)
# result.status       → VisibilityStatus.TOO_CLOSE | GOOD | TOO_FAR
# result.is_ok        → True/False
# result.edge_count   → int raw edge pixel count
```

**Recommended thresholds:**

| Threshold | Value | Meaning |
|---|---|---|
| `DENSITY_CLOSE_THRESHOLD` | 0.18 | Above → Move Further Away |
| `DENSITY_FAR_THRESHOLD` | 0.03 | Below → Move Closer |

### Quality Orchestrator (`quality_analyzer.py`)

Single entry point that runs all four analyzers and returns a unified `QualityReport`:

```python
from preprocessing.quality_analyzer import analyze_frame, analyze_pil_image

# From webcam frame (BGR numpy array)
report = analyze_frame(frame)

# From uploaded PIL image
report = analyze_pil_image(pil_image)

# With custom thresholds
report = analyze_frame(frame, blur_clear=150, tilt_threshold=5.0, dark_threshold=70)

# Use the report
print(report.quality_pct)     # 0, 25, 50, 75, or 100
print(report.overall_ok)      # True only when ALL 4 pass
print(report.analysis_time_ms) # e.g., 12.4

# Phase D gate
if report.overall_ok:
    cells = braille_detector.detect(frame)  # ← Phase D
```

### Guidance Panel (`ui/guidance_panel.py`)

Premium hackathon-ready Streamlit panel with:
- **4 metric cards** with animated progress bars and colour-coded status
- **Tilt needle indicator** — visual compass for page orientation
- **AI Guidance Board** — prioritised corrective tips with HIGH/MEDIUM labels
- **Overall score widget** — large quality percentage with qualitative label
- **Phase D gate indicator** — clear OPEN / LOCKED signal

```python
from ui.guidance_panel import render_full_guidance_panel
render_full_guidance_panel(report)   # renders in current Streamlit column
```

---

## 📁 Full Folder Structure

```
BrailleVisionAI/
│
├── app.py                        ← Streamlit entry point (Phase B + C)
│
├── preprocessing/
│   ├── __init__.py
│   ├── blur_detector.py          ← Phase C: Laplacian variance blur check
│   ├── brightness_analyzer.py    ← Phase C: Brightness + contrast analysis
│   ├── alignment_checker.py      ← Phase C: Hough line tilt detection
│   ├── visibility_estimator.py   ← Phase C: Edge density distance proxy
│   ├── quality_analyzer.py       ← Phase C: Orchestrator (single entry point)
│   └── preprocess.py             ← Phase D placeholder
│
├── ui/
│   ├── __init__.py
│   ├── guidance_panel.py         ← Phase C: Full quality dashboard UI
│   ├── camera_ui.py              ← Phase B: Webcam management + overlays
│   └── interface.py              ← Phase B: Additional UI components
│
├── detection/                    ← Phase D placeholder
├── translation/                  ← Phase E placeholder
├── speech/                       ← Phase F placeholder
├── datasets/
│   ├── captured_frames/          ← Snapshot images from webcam
│   └── sample_test_images/       ← Uploaded test images
│
├── models/                       ← Phase D: YOLO model weights
├── notebooks/                    ← Jupyter R&D notebooks
├── tests/                        ← Unit & integration tests
├── screenshots/                  ← App screenshots for documentation
├── demo/                         ← Demo videos & GIFs
├── docs/                         ← Additional documentation
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip package manager
- A webcam (for live quality analysis)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/BrailleVisionAI.git
cd BrailleVisionAI
```

### 2. Create & Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Minimum for Phase C (fast install):**
```bash
pip install streamlit>=1.32.0 opencv-python>=4.9.0 Pillow>=10.2.0 numpy>=1.26.0
```

**Full install (all phases):**
```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🧪 Testing Phase C

### Run all Phase C analyzers on a test image

```bash
python -c "
import cv2, numpy as np
from preprocessing.quality_analyzer import analyze_frame
frame = cv2.imread('datasets/sample_test_images/your_image.jpg')
if frame is None:
    # Use a synthetic frame for testing
    frame = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
report = analyze_frame(frame)
print(f'Quality: {report.quality_pct}% ({report.ok_count}/4 checks)')
print(f'Blur:    {report.blur.score:.1f} — {report.blur.status.value}')
print(f'Bright:  {report.brightness.score:.1f} — {report.brightness.status.value}')
print(f'Align:   {report.alignment.angle}° — {report.alignment.status.value}')
print(f'Dist:    {report.visibility.edge_density:.4f} — {report.visibility.status.value}')
print(f'Ready for Phase D: {report.overall_ok}')
"
```

### Expected output (for a well-lit, in-focus Braille page)

```
Quality: 100% (4/4 checks)
Blur:    243.7 — Clear Image
Bright:  138.2 — Good Lighting
Align:   1.4° — Properly Aligned
Dist:    0.0823 — Good Distance
Ready for Phase D: True
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

---

## 🎛️ Sample Threshold Recommendations

These starting values work well for a standard USB webcam at ~25 cm from a Braille page under desk lamp lighting:

```python
# Blur (Laplacian variance — higher = sharper)
BLUR_CLEAR_THRESHOLD    = 120.0   # above → Clear Image
BLUR_SLIGHT_THRESHOLD   = 60.0    # 60–120 → Slightly Blurry

# Brightness (mean pixel value 0–255)
BRIGHT_DARK_THRESHOLD   = 60      # below → Too Dark
BRIGHT_OVER_THRESHOLD   = 210     # above → Overexposed
CONTRAST_MIN            = 20      # std-dev below → Low Contrast

# Alignment (degrees from horizontal)
TILT_THRESHOLD          = 8.0     # |angle| above this → Tilted

# Visibility (Canny edge density ratio 0.0–1.0)
DENSITY_CLOSE_THRESHOLD = 0.18    # above → Too Close
DENSITY_FAR_THRESHOLD   = 0.03    # below → Too Far
```

Fine-tune via the **⚙️ Thresholds** page inside the running app.

---

## 🔭 Future Scope

- **Phase D**: Image preprocessing (denoise, CLAHE) + YOLOv8 Braille dot detection
- **Phase E**: Braille cell pattern → Grade-1 English translation
- **Phase F**: Text-to-speech voice output (pyttsx3 / gTTS)
- **Grade-2 Braille Support** — Contracted Braille with abbreviations
- **Real-Time Mobile App** — Flutter/React Native frontend
- **Edge Deployment** — Raspberry Pi / Jetson Nano
- **Cloud API** — REST API for third-party accessibility tools

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🏆 Hackathon

Built for the **AI for Accessibility** track.

> *"Technology should be accessible to everyone — this is our contribution to making that vision real."*
