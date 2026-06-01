"""
============================================================
BrailleVisionAI — Phase D  |  Braille Presence Detector
detection/braille_detector.py
============================================================

PURPOSE
-------
Main Phase D module.  Determines whether a camera frame or uploaded
image contains Braille writing, using a HYBRID approach:

  LAYER 1 — OpenCV Heuristics  (fast, runs on every frame)
    └─ BrailleHeuristics.analyze() in detection/heuristics.py

  LAYER 2 — AI Classifier  (slower, runs when heuristics pass)
    └─ Trained YOLOv8 classification model
       OR MobileNetV2-based binary classifier (braille / non_braille)
       OR fallback: heuristics-only when no model file present

DECISION LOGIC
--------------
  heuristic_score + ai_confidence → final_label + final_confidence

  Label categories:
    "Braille Detected"   → final_confidence ≥ HIGH_THRESHOLD (0.70)
    "Possibly Braille"   → final_confidence ≥ LOW_THRESHOLD  (0.45)
    "No Braille"         → final_confidence <  LOW_THRESHOLD

FUSION STRATEGY
---------------
  If AI model available:
    final = 0.40 × heuristic_score + 0.60 × ai_confidence

  If NO model (heuristics-only mode):
    final = heuristic_score  (with a 15% confidence reduction to signal uncertainty)

HOW TO USE
----------
    from detection.braille_detector import BraillePresenceDetector, DetectionResult

    detector = BraillePresenceDetector()          # auto-finds model if present
    result   = detector.detect(bgr_frame)

    print(result.label)           # "Braille Detected" / "Possibly Braille" / "No Braille"
    print(result.confidence)      # e.g. 0.87
    print(result.confidence_pct)  # e.g. "87%"
    print(result.dot_count)       # number of blobs detected
    print(result.is_braille)      # True if label == "Braille Detected"
    print(result.is_uncertain)    # True if label == "Possibly Braille"
    print(result.annotated_frame) # BGR frame with overlays

PHASE E HOOK
------------
    if result.is_braille:
        cells = cell_recognizer.recognize(bgr_frame)  # ← Phase E
============================================================
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from detection.heuristics import BrailleHeuristics, HeuristicResult


# ── Thresholds ──────────────────────────────────────────────
HIGH_THRESHOLD = 0.70   # ≥ this → "Braille Detected"
LOW_THRESHOLD  = 0.40   # ≥ this → "Possibly Braille"

# Default path where the trained model is expected
DEFAULT_MODEL_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "braille_presence",
)
YOLO_MODEL_PATH    = os.path.join(DEFAULT_MODEL_DIR, "braille_yolov8_cls.pt")
KERAS_MODEL_PATH   = os.path.join(DEFAULT_MODEL_DIR, "braille_classifier.keras")


# ── Detection result dataclass ───────────────────────────────
@dataclass
class DetectionResult:
    """
    Full output of the Braille presence detection pipeline.

    Attributes:
        label:          "Braille Detected" | "Possibly Braille" | "No Braille"
        confidence:     Final fused confidence in [0.0, 1.0].
        heuristic_score: Raw score from the OpenCV heuristic layer.
        ai_confidence:  Score from the AI model layer (None if model absent).
        dot_count:      Number of circular blobs found by heuristics.
        row_count:      Number of horizontal rows detected.
        avg_spacing:    Mean nearest-neighbour spacing between dots.
        dot_centers:    List of (x, y) for each detected dot.
        annotated_frame: BGR image with detection overlays.
        model_used:     "yolov8" | "keras" | "heuristics_only"
    """
    label:           str   = "No Braille"
    confidence:      float = 0.0
    heuristic_score: float = 0.0
    ai_confidence:   Optional[float] = None
    dot_count:       int   = 0
    row_count:       int   = 0
    avg_spacing:     float = 0.0
    dot_centers:     list  = field(default_factory=list)
    annotated_frame: Optional[np.ndarray] = field(default=None, repr=False)
    model_used:      str   = "heuristics_only"
    heuristic_result: Optional[object]   = field(default=None, repr=False)  # HeuristicResult v4

    @property
    def confidence_pct(self) -> str:
        """Formatted confidence percentage string, e.g. '87%'."""
        return f"{int(self.confidence * 100)}%"

    @property
    def is_braille(self) -> bool:
        return self.label == "Braille Detected"

    @property
    def is_uncertain(self) -> bool:
        return self.label == "Possibly Braille"

    @property
    def is_no_braille(self) -> bool:
        return self.label == "No Braille"

    @property
    def status_color(self) -> Tuple[int, int, int]:
        """BGR color for OpenCV overlays: green / yellow / red."""
        if self.is_braille:   return (0, 220, 60)    # green
        if self.is_uncertain: return (0, 180, 255)   # amber
        return (0, 60, 220)                           # red


# ── Main detector class ──────────────────────────────────────
class BraillePresenceDetector:
    """
    Hybrid Braille presence detector.

    Loads a YOLO or Keras classifier from disk if available.
    Falls back to heuristics-only mode when no model file is found.

    Args:
        model_dir:       Directory containing trained model weights.
        heuristic_only:  Force heuristic-only mode even if model present.
        high_threshold:  Confidence threshold for "Braille Detected".
        low_threshold:   Confidence threshold for "Possibly Braille".
    """

    def __init__(
        self,
        model_dir:       str   = DEFAULT_MODEL_DIR,
        heuristic_only:  bool  = False,
        high_threshold:  float = HIGH_THRESHOLD,
        low_threshold:   float = LOW_THRESHOLD,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold  = low_threshold
        self.heuristics     = BrailleHeuristics()

        self._yolo_model    = None
        self._keras_model   = None
        self._model_type    = "heuristics_only"

        if not heuristic_only:
            self._try_load_model(model_dir)

    # ── Model loading ────────────────────────────────────────
    def _try_load_model(self, model_dir: str) -> None:
        """
        Attempt to load a trained model.  Tries YOLO first, then Keras.
        Sets self._model_type accordingly.  Silently continues if no model found.
        """
        yolo_path  = os.path.join(model_dir, "braille_yolov8_cls.pt")
        keras_path = os.path.join(model_dir, "braille_classifier.keras")

        # ── Try YOLOv8 classification model ──────────────────
        if os.path.exists(yolo_path):
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(yolo_path)
                self._model_type = "yolov8"
                print(f"[BrailleDetector] OK Loaded YOLO model: {yolo_path}")
                return
            except Exception as e:
                print(f"[BrailleDetector] WARN YOLO load failed: {e}")

        # ── Try Keras / TF classifier ─────────────────────────
        if os.path.exists(keras_path):
            try:
                from keras.models import load_model as keras_load
                self._keras_model = keras_load(keras_path, compile=False)
                self._model_type  = "keras"
                print(f"[BrailleDetector] OK Loaded Keras model: {keras_path}")
                return
            except Exception as e:
                print(f"[BrailleDetector] WARN Keras load failed: {e}")

        print(
            "[BrailleDetector] INFO No trained model found - running in "
            "heuristics-only mode.  Train a model and place it in "
            f"'{model_dir}' to enable AI classification."
        )

    # ── AI inference ─────────────────────────────────────────
    def _yolo_inference(self, bgr: np.ndarray) -> Optional[float]:
        """
        Run YOLOv8 classification on the frame.

        Returns:
            Probability of class "braille" in [0.0, 1.0], or None on error.
        """
        try:
            results = self._yolo_model.predict(bgr, verbose=False)
            probs   = results[0].probs   # ClassificationResult.probs
            # Class 0 = braille, class 1 = non_braille (set during training)
            braille_prob = float(probs.data[0])
            return braille_prob
        except Exception as e:
            print(f"[BrailleDetector] YOLO inference error: {e}")
            return None

    def _keras_inference(self, bgr: np.ndarray) -> Optional[float]:
        """
        Run Keras binary classifier on the frame.

        Preprocessing:
            BGR → RGB, resize to 224×224, normalise to [0, 1],
            add batch dimension → predict → class 0 = braille probability.

        Returns:
            Probability of "braille" class in [0.0, 1.0], or None on error.
        """
        try:
            import numpy as np
            rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (224, 224))
            x       = resized.astype("float32") / 255.0
            x       = np.expand_dims(x, axis=0)   # (1, 224, 224, 3)
            pred    = self._keras_model.predict(x, verbose=0)
            # pred shape: (1, 1) for binary sigmoid or (1, 2) for softmax
            if pred.shape[-1] == 1:
                return float(pred[0, 0])
            else:
                return float(pred[0, 0])   # index 0 = braille class
        except Exception as e:
            print(f"[BrailleDetector] Keras inference error: {e}")
            return None

    # ── Score fusion ──────────────────────────────────────────
    def _fuse(self, heuristic_score: float, ai_confidence: Optional[float]) -> float:
        """
        Fuse heuristic score with AI confidence into a single value.

        Weights:
            With AI model:      60% AI  + 40% heuristic
            Without AI model:   100% heuristic (reduced by 15% to signal uncertainty)
        """
        if ai_confidence is not None:
            return 0.60 * ai_confidence + 0.40 * heuristic_score
        else:
            return heuristic_score * 0.85

    # ── Label from confidence ─────────────────────────────────
    def _label(self, confidence: float) -> str:
        if confidence >= self.high_threshold:
            return "Braille Detected"
        elif confidence >= self.low_threshold:
            return "Possibly Braille"
        else:
            return "No Braille"

    # ── Main API ─────────────────────────────────────────────
    def detect(
        self,
        bgr:            np.ndarray,
        detection_mode: str = "balanced",
    ) -> DetectionResult:
        """
        Run the full Braille presence detection pipeline on a BGR frame.

        Args:
            bgr:            BGR numpy array (webcam frame or PIL→numpy).
            detection_mode: 'relaxed' | 'balanced' | 'strict'

        Returns:
            DetectionResult with label, confidence, annotated frame,
            and heuristic_result (contains debug stats for panel).
        """
        result = DetectionResult()
        result.model_used = self._model_type

        # ── Layer 1: Heuristics ──────────────────────────────
        h_result: HeuristicResult = self.heuristics.analyze(
            bgr, detection_mode=detection_mode
        )
        result.heuristic_score  = h_result.score
        result.dot_count        = h_result.dot_count
        result.row_count        = h_result.row_count
        result.avg_spacing      = h_result.avg_spacing
        result.dot_centers      = h_result.dot_centers
        result.heuristic_result = h_result   # ← exposes debug stats

        annotated = h_result.annotated_frame if h_result.annotated_frame is not None else bgr.copy()

        # ── Layer 2: AI inference ────────────────────────────
        ai_confidence = None
        if h_result.score >= 0.15:
            if self._model_type == "yolov8" and self._yolo_model:
                ai_confidence = self._yolo_inference(bgr)
            elif self._model_type == "keras" and self._keras_model:
                ai_confidence = self._keras_inference(bgr)
        result.ai_confidence = ai_confidence

        # ── Layer 3: Fuse scores ─────────────────────────────
        final_conf        = self._fuse(h_result.score, ai_confidence)
        result.confidence = round(min(1.0, max(0.0, final_conf)), 3)
        result.label      = self._label(result.confidence)

        # ── Layer 4: Annotate frame ──────────────────────────
        result.annotated_frame = self._draw_detection_overlay(annotated, result)

        return result

    # ── Frame annotation ──────────────────────────────────────
    def _draw_detection_overlay(
        self,
        frame: np.ndarray,
        result: "DetectionResult",
    ) -> np.ndarray:
        """
        Draw detection status and confidence on the frame.

        Draws:
            - Coloured border rectangle around full frame
            - Label + confidence badge in top-left corner
            - Dot count info in top-right corner
        """
        out    = frame.copy()
        h, w   = out.shape[:2]
        color  = result.status_color   # BGR

        # ── Border rectangle ─────────────────────────────────
        thickness = 4 if result.is_braille else 2
        cv2.rectangle(out, (2, 2), (w - 2, h - 2), color, thickness)

        # ── Label badge (top-left) ────────────────────────────
        label_text = f"{result.label}  {result.confidence_pct}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_DUPLEX, 0.65, 1
        )
        pad = 8
        cv2.rectangle(out, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2 + baseline), color, -1)
        cv2.putText(
            out, label_text,
            (8 + pad, 8 + pad + th),
            cv2.FONT_HERSHEY_DUPLEX, 0.65,
            (0, 0, 0), 1, cv2.LINE_AA,
        )

        # ── Dot info badge (top-right) ────────────────────────
        info = f"Dots:{result.dot_count}  Rows:{result.row_count}"
        (iw, ih), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ix = w - iw - 16
        cv2.rectangle(out, (ix - 4, 8), (w - 8, 8 + ih + 14), (20, 20, 20), -1)
        cv2.putText(
            out, info,
            (ix, 8 + ih + 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52,
            (200, 200, 200), 1, cv2.LINE_AA,
        )

        # ── Model badge (bottom-left) ─────────────────────────
        model_lbl = f"Mode: {result.model_used}"
        cv2.putText(
            out, model_lbl,
            (10, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.44,
            (120, 120, 120), 1, cv2.LINE_AA,
        )

        return out
