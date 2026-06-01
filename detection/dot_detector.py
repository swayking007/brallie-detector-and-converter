"""
============================================================
BrailleVisionAI - Phase H.5  |  Embossed Dot Detector (v8.1)
detection/dot_detector.py
============================================================

PHASE H.5 PATCHED: RECALL-FIRST DETECTION
-------------------------------------------
v8.1 patches the overly strict v8 gates that dropped real dots.
Philosophy: MISSING REAL DOTS IS WORSE THAN ALLOWING EXTRA CANDIDATES.

  STEP 1 - Preprocessing
    Grayscale -> CLAHE(2.5) -> bilateralFilter(d=9, s=75)
    -> adaptiveThreshold -> MORPH_OPEN -> MORPH_CLOSE

  STEP 2 - Relaxed shape gates (wide to maximise recall)
    10 < area < 800
    0.35 < circularity < 1.4
    3px < radius < 25px
    0.50 < aspect_ratio < 1.50

  STEP 2b - Local contrast gate (center vs ring)
    contrast = abs(center_intensity - ring_intensity)
    keep only if contrast > 12

  STEP 2c - Non-maximum suppression
    If two candidates within distance < median_radius * 1.5
    keep only the stronger one.

  STEP 2d - Dynamic radius filtering
    Compute median radius from candidates.
    Reject: radius < median*0.5  or  radius > median*2.0

  STEPS 3-5 - Geometric Grid Engine
    Ghost grid, slot snapping, confidence scoring.

  STEP 6 - Visual debug panel

Detection modes:  'relaxed' | 'balanced' | 'strict'
"""

from __future__ import annotations

import os
import math
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from detection.braille_pattern import BrailleDot
from detection.grid_engine import (
    BrailleGridEngine,
    GridCell,
    draw_grid_debug,
)

# ── Model path ───────────────────────────────────────────────
DEFAULT_DOT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "braille_dots", "braille_dots_yolov8.pt"
)

MAX_PROCESS_WIDTH   = 1280   # resize if wider
MAX_CANDIDATES      = 200    # cap before grid engine (raised for recall)
DEDUP_RADIUS_FACTOR = 0.38

# ── STEP 2: Recall-first candidate filter constants ──────────
# PATCHED: Relaxed gates to prioritise recall over precision.
# Missing real dots is worse than allowing extra candidates.
# The grid engine (STEP 3-5) handles false-positive suppression.
BASE_AREA_MIN   = 10
BASE_AREA_MAX   = 800
BASE_CIRC_MIN   = 0.35
BASE_CIRC_MAX   = 1.4
BASE_RADIUS_MIN = 3.0     # px
BASE_RADIUS_MAX = 25.0    # px
BASE_ASPECT_MIN = 0.50
BASE_ASPECT_MAX = 1.50

# Local contrast gate: center vs surrounding ring
CONTRAST_THRESHOLD = 12   # keep if abs(center - ring) > this

# Dynamic radius filtering: reject outliers vs median radius
DYN_RADIUS_LO = 0.5    # keep if radius >= median * this
DYN_RADIUS_HI = 2.0    # keep if radius <= median * this

# NMS: suppress weaker candidate within this factor of dot radius
NMS_RADIUS_FACTOR = 1.5

# ── Per-mode overrides ───────────────────────────────────────
_MODE: Dict[str, Dict] = {
    "relaxed": dict(
        area_min=8,    area_max=1000,
        circ_min=0.30, circ_max=1.5,
        r_min=2.5,     r_max=28.0,
        asp_min=0.45,  asp_max=1.55,
    ),
    "balanced": dict(
        area_min=BASE_AREA_MIN, area_max=BASE_AREA_MAX,
        circ_min=BASE_CIRC_MIN, circ_max=BASE_CIRC_MAX,
        r_min=BASE_RADIUS_MIN,  r_max=BASE_RADIUS_MAX,
        asp_min=BASE_ASPECT_MIN, asp_max=BASE_ASPECT_MAX,
    ),
    "strict": dict(
        area_min=18,   area_max=500,
        circ_min=0.50, circ_max=1.3,
        r_min=3.5,     r_max=22.0,
        asp_min=0.65,  asp_max=1.35,
    ),
}

# ── Singleton grid engine ─────────────────────────────────────
_GRID_ENGINE = BrailleGridEngine()


class BrailleDotDetector:
    """
    Phase H.5 geometry-constrained Braille dot detector.

    Public API
    ----------
    detect(bgr, avg_spacing, detect_mode, demo_mode)
        → List[BrailleDot]
    detect_with_debug(bgr, avg_spacing, detect_mode, demo_mode)
        → (accepted, rejected, debug_frame, stats_dict)
    """

    def __init__(self, model_path: str = DEFAULT_DOT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.yolo_model = None
        self.mode       = "opencv"
        self._try_load_model()

    def _try_load_model(self) -> None:
        if os.path.exists(self.model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(self.model_path)
                self.mode = "yolov8"
                print(f"[DotDetector] OK YOLOv8 loaded: {self.model_path}")
            except Exception as e:
                print(f"[DotDetector] WARN YOLO load failed: {e}")
        else:
            print("[DotDetector] INFO No YOLO model -> OpenCV H.5 geometry pipeline.")

    # ── Public ───────────────────────────────────────────────
    def detect(
        self,
        bgr_frame:   np.ndarray,
        avg_spacing: float = 15.0,
        detect_mode: str   = "balanced",
        demo_mode:   bool  = False,
    ) -> List[BrailleDot]:
        if self.mode == "yolov8" and self.yolo_model:
            return self._detect_yolo(bgr_frame)
        result = self._pipeline(bgr_frame, avg_spacing, detect_mode, demo_mode)
        return result[0]

    def detect_with_debug(
        self,
        bgr_frame:   np.ndarray,
        avg_spacing: float = 15.0,
        detect_mode: str   = "balanced",
        demo_mode:   bool  = False,
    ) -> Tuple[List[BrailleDot], List[BrailleDot], np.ndarray, Dict]:
        if self.mode == "yolov8" and self.yolo_model:
            accepted = self._detect_yolo(bgr_frame)
            empty    = {"raw_contour_count": 0, "filtered_count": len(accepted),
                        "rejected_tiny": 0, "rejected_irregular": 0,
                        "rejected_size": 0, "processing_ms": 0.0,
                        "geo_conf": 0.0, "ghost_count": 0}
            return accepted, [], bgr_frame.copy(), empty

        accepted, rejected, stats, dbg_data = self._pipeline(
            bgr_frame, avg_spacing, detect_mode, demo_mode
        )
        debug_frame = draw_grid_debug(
            bgr_frame, accepted, rejected,
            dbg_data.get("ghost_dots", []),
            dbg_data.get("grid_cells", []),
            angle=_GRID_ENGINE.last_angle,
        )
        return accepted, rejected, debug_frame, stats

    # ── YOLO fallback ────────────────────────────────────────
    def _detect_yolo(self, bgr: np.ndarray) -> List[BrailleDot]:
        dots: List[BrailleDot] = []
        try:
            res = self.yolo_model.predict(bgr, verbose=False)
            for box in res[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cx   = int((xyxy[0] + xyxy[2]) / 2)
                cy   = int((xyxy[1] + xyxy[3]) / 2)
                r    = float(((xyxy[2]-xyxy[0])+(xyxy[3]-xyxy[1]))/4)
                dots.append(BrailleDot(x=cx, y=cy, radius=r, confidence=conf))
        except Exception as e:
            print(f"[DotDetector] YOLO error: {e}")
            return self._pipeline(bgr)[0]
        return dots

    # ══════════════════════════════════════════════════════════
    # STEP 1 — IMAGE PREPROCESSING
    # ══════════════════════════════════════════════════════════
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Geometry-constrained preprocessing:
          1. Grayscale
          2. CLAHE(clipLimit=2.5)
          3. bilateralFilter(d=9, σColor=75, σSpace=75)
          4. adaptiveThreshold
          5. morphologyEx MORPH_OPEN (remove tiny artifacts)
          6. morphologyEx MORPH_CLOSE (bridge small gaps)

        Returns: (enhanced_gray, binary_mask, gray)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()

        # CLAHE — enhance embossed dot contrast
        clahe    = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Bilateral filter — preserve edges, remove paper texture noise
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # Adaptive threshold — extract dark/raised regions
        binary = cv2.adaptiveThreshold(
            bilateral, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=19, C=5,
        )

        k3 = np.ones((3, 3), np.uint8)

        # MORPH_OPEN — remove tiny artifacts (paper grain, sensor noise)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)

        # MORPH_CLOSE — bridge small gaps in embossed dots
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k3, iterations=1)

        return enhanced, binary, gray

    # ══════════════════════════════════════════════════════════
    # MAIN H.5 PIPELINE
    # ══════════════════════════════════════════════════════════
    def _pipeline(
        self,
        bgr_frame:   np.ndarray,
        avg_spacing: float = 15.0,
        detect_mode: str   = "balanced",
        demo_mode:   bool  = False,
    ) -> Tuple[List[BrailleDot], List[BrailleDot], Dict, Dict]:
        t0 = time.perf_counter()

        if demo_mode and detect_mode == "relaxed":
            detect_mode = "balanced"
        p = _MODE.get(detect_mode, _MODE["balanced"])

        # ── Resize ───────────────────────────────────────────
        orig_h, orig_w = bgr_frame.shape[:2]
        scale = 1.0
        frame = bgr_frame
        if orig_w > MAX_PROCESS_WIDTH:
            scale = MAX_PROCESS_WIDTH / orig_w
            frame = cv2.resize(bgr_frame,
                               (MAX_PROCESS_WIDTH, int(orig_h * scale)),
                               interpolation=cv2.INTER_AREA)

        proc_h, proc_w = frame.shape[:2]

        # ── STEP 1: Preprocessing ────────────────────────────
        enhanced, binary, gray = self._preprocess(frame)

        # ── STEP 2: Recall-first dot candidate filtering ─────
        raw_candidates, rejected_by_shape = self._extract_candidates(
            binary, enhanced, p, scale
        )

        # ── Local contrast gate ──────────────────────────────
        contrast_passed = []
        for dot in raw_candidates:
            cs = _center_ring_contrast(enhanced, dot.x, dot.y, max(3, int(dot.radius)))
            if cs > CONTRAST_THRESHOLD:
                contrast_passed.append(dot)
            else:
                rejected_by_shape.append(dot)

        # ── NMS: radius-adaptive non-maximum suppression ─────
        if contrast_passed:
            radii = [d.radius for d in contrast_passed]
            est_r = float(np.median(radii)) if radii else 8.0
            nms_dist = est_r * NMS_RADIUS_FACTOR
            nms_passed = _deduplicate(contrast_passed, nms_dist)
        else:
            nms_passed = []

        # ── Dynamic radius filtering ─────────────────────────
        if len(nms_passed) >= 3:
            med_r = float(np.median([d.radius for d in nms_passed]))
            radius_filtered = []
            for d in nms_passed:
                if med_r * DYN_RADIUS_LO <= d.radius <= med_r * DYN_RADIUS_HI:
                    radius_filtered.append(d)
                else:
                    rejected_by_shape.append(d)
            raw_deduped = radius_filtered
        else:
            raw_deduped = nms_passed

        if len(raw_deduped) > MAX_CANDIDATES:
            raw_deduped = sorted(raw_deduped,
                                 key=lambda d: d.confidence,
                                 reverse=True)[:MAX_CANDIDATES]

        # ── STEPS 3-5: Geometric Grid Engine ─────────────────
        confirmed_dots, rejected_by_grid, grid_cells, geo_conf = _GRID_ENGINE.process(
            raw_deduped, enhanced, proc_w, proc_h, avg_spacing * scale
        )

        accepted = confirmed_dots

        # Combine all rejected dots
        all_rejected = rejected_by_shape + rejected_by_grid

        # Collect ghost/recovered dots from grid cells
        ghost_dots = []
        for gc in grid_cells:
            for s_idx in range(6):
                if gc.ghost[s_idx] and gc.confirmed[s_idx] is not None:
                    ghost_dots.append(gc.confirmed[s_idx])

        # Rescale to original image coords
        if scale != 1.0:
            accepted     = _rescale_dots(accepted,     scale)
            ghost_dots   = _rescale_dots(ghost_dots,   scale)
            all_rejected = _rescale_dots(all_rejected,  scale)

        ms = (time.perf_counter() - t0) * 1000

        stats = {
            "raw_contour_count":  len(raw_deduped) + len(rejected_by_shape),
            "filtered_count":     len(accepted),
            "rejected_tiny":      len(rejected_by_shape),
            "rejected_irregular": len(rejected_by_grid),
            "rejected_size":      0,
            "processing_ms":      round(ms, 1),
            "geo_conf":           round(geo_conf, 3),
            "ghost_count":        len(ghost_dots),
            "spacing":            _GRID_ENGINE.last_spacing,
            "angle":              _GRID_ENGINE.last_angle,
        }

        dbg_data = {
            "enhanced":   enhanced,
            "binary":     binary,
            "grid_cells": grid_cells,
            "ghost_dots": ghost_dots,
        }

        return accepted, all_rejected, stats, dbg_data

    # ══════════════════════════════════════════════════════════
    # STEP 2 — RECALL-FIRST CANDIDATE EXTRACTION
    # ══════════════════════════════════════════════════════════
    def _extract_candidates(
        self,
        binary:   np.ndarray,
        enhanced: np.ndarray,
        p:        dict,
        scale:    float,
    ) -> Tuple[List[BrailleDot], List[BrailleDot]]:
        """
        Extract dot candidates with relaxed shape gates.

        For each contour compute:
          - Area
          - Circularity = 4*pi*area / perimeter^2
          - Radius (min enclosing circle)
          - Aspect ratio = width / height (bounding rect)

        Gates are wide to prioritise recall.
        Local contrast + NMS + dynamic radius handle false positives later.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        candidates: List[BrailleDot] = []
        rejected:   List[BrailleDot] = []
        h, w = enhanced.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # ── Gate 1: Area ─────────────────────────────────
            if not (p["area_min"] < area < p["area_max"]):
                if area > 3:
                    M = cv2.moments(cnt)
                    if M["m00"] > 0:
                        rejected.append(BrailleDot(
                            x=int(int(M["m10"] / M["m00"]) / scale),
                            y=int(int(M["m01"] / M["m00"]) / scale),
                            radius=3.0, confidence=0.0))
                continue

            # ── Gate 2: Circularity ──────────────────────────
            perim = cv2.arcLength(cnt, True)
            if perim < 1:
                continue
            circularity = (4 * math.pi * area) / (perim * perim)
            if not (p["circ_min"] < circularity < p["circ_max"]):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    rejected.append(BrailleDot(
                        x=int(int(M["m10"] / M["m00"]) / scale),
                        y=int(int(M["m01"] / M["m00"]) / scale),
                        radius=3.0, confidence=0.0))
                continue

            # ── Gate 3: Radius ───────────────────────────────
            (cx_f, cy_f), enc_r = cv2.minEnclosingCircle(cnt)
            if not (p["r_min"] < enc_r < p["r_max"]):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    rejected.append(BrailleDot(
                        x=int(int(M["m10"] / M["m00"]) / scale),
                        y=int(int(M["m01"] / M["m00"]) / scale),
                        radius=enc_r / scale, confidence=0.0))
                continue

            # ── Gate 4: Aspect ratio ─────────────────────────
            x_r, y_r, bw, bh = cv2.boundingRect(cnt)
            if bh < 1:
                continue
            aspect = bw / bh
            if not (p["asp_min"] < aspect < p["asp_max"]):
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    rejected.append(BrailleDot(
                        x=int(int(M["m10"] / M["m00"]) / scale),
                        y=int(int(M["m01"] / M["m00"]) / scale),
                        radius=enc_r / scale, confidence=0.0))
                continue

            # ── All gates passed — compute centroid + confidence ─
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx_s = int(M["m10"] / M["m00"])
            cy_s = int(M["m01"] / M["m00"])

            # Dot quality score: blend of shape + local contrast
            circ_score   = float(np.clip((circularity - p["circ_min"]) /
                                         (1.0 - p["circ_min"] + 1e-6), 0, 1))
            aspect_score = 1.0 - abs(aspect - 1.0)
            contrast_raw = _center_ring_contrast(enhanced, cx_s, cy_s, max(3, int(enc_r)))
            contrast_s   = float(np.clip(contrast_raw / 40.0, 0.0, 1.0))
            quality      = float(np.clip(
                0.3 * circ_score + 0.25 * aspect_score + 0.45 * contrast_s,
                0.0, 1.0
            ))

            real_r = enc_r / scale
            cx_out = int(cx_s / scale)
            cy_out = int(cy_s / scale)
            candidates.append(BrailleDot(
                x=cx_out, y=cy_out,
                radius=real_r, confidence=quality
            ))

        return candidates, rejected


# ── Helpers ───────────────────────────────────────────────────

def _center_ring_contrast(
    gray: np.ndarray, cx: int, cy: int, r: int
) -> float:
    """
    Local contrast: abs(center_intensity - surrounding_ring_intensity).

    Real embossed dots have a measurable intensity difference between
    the dot centre and the surrounding paper.  Paper texture does not.

    Returns raw absolute difference (NOT normalised).
    Caller should check: contrast_score > CONTRAST_THRESHOLD (12).
    """
    h, w = gray.shape
    r = max(3, r)

    # Center patch: inner circle region
    cx1 = max(0, cx - r); cy1 = max(0, cy - r)
    cx2 = min(w, cx + r); cy2 = min(h, cy + r)
    center_patch = gray[cy1:cy2, cx1:cx2]
    if center_patch.size == 0:
        return 0.0
    center_mean = float(np.mean(center_patch))

    # Surrounding ring: annulus from r to 2*r
    r_out = r * 2
    rx1 = max(0, cx - r_out); ry1 = max(0, cy - r_out)
    rx2 = min(w, cx + r_out); ry2 = min(h, cy + r_out)
    ring_patch = gray[ry1:ry2, rx1:rx2]
    if ring_patch.size == 0:
        return 0.0

    # Create annular mask: outside center, inside outer ring
    ring_h, ring_w = ring_patch.shape
    yy, xx = np.ogrid[:ring_h, :ring_w]
    # Centre of ring_patch in local coords
    local_cx = cx - rx1
    local_cy = cy - ry1
    dist_sq = (xx - local_cx) ** 2 + (yy - local_cy) ** 2
    inner_mask = dist_sq <= r * r
    outer_mask = dist_sq <= r_out * r_out
    annulus_mask = outer_mask & ~inner_mask

    ring_pixels = ring_patch[annulus_mask]
    if ring_pixels.size == 0:
        return abs(center_mean - float(np.mean(ring_patch)))

    ring_mean = float(np.mean(ring_pixels))
    return abs(center_mean - ring_mean)


# Legacy alias for backwards compatibility (validate_refactor.py)
_local_contrast = _center_ring_contrast


def _deduplicate(dots: List[BrailleDot], min_dist: float) -> List[BrailleDot]:
    """Merge nearby candidates by keeping highest-confidence one."""
    if not dots:
        return []
    dots_s = sorted(dots, key=lambda d: d.confidence, reverse=True)
    kept   = []
    supp   = [False] * len(dots_s)
    for i, dot in enumerate(dots_s):
        if supp[i]:
            continue
        kept.append(dot)
        for j in range(i + 1, len(dots_s)):
            if supp[j]:
                continue
            dx = dot.x - dots_s[j].x
            dy = dot.y - dots_s[j].y
            if dx*dx + dy*dy < min_dist * min_dist:
                supp[j] = True
    return kept


def _rescale_dots(dots: List[BrailleDot], scale: float) -> List[BrailleDot]:
    """Convert coordinates from processing scale back to original image scale."""
    if scale == 1.0:
        return dots
    inv = 1.0 / scale
    return [
        BrailleDot(
            x=int(d.x * inv),
            y=int(d.y * inv),
            radius=d.radius * inv,
            confidence=d.confidence,
        )
        for d in dots
    ]
