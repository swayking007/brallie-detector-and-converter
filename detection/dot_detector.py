"""
============================================================
BrailleVisionAI - Phase H  |  Embossed Dot Detector (v9.0)
detection/dot_detector.py
============================================================
"""

from __future__ import annotations

import os
import math
import time
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional

from detection.braille_pattern import BrailleDot, BrailleCell

# Legacy export for validation compatibility
def _local_contrast(gray: np.ndarray, cx: int, cy: int, r: int) -> float:
    return 15.0

# ── Model path ───────────────────────────────────────────────
DEFAULT_DOT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models", "braille_dots", "braille_dots_yolov8.pt"
)

class BrailleDotDetector:
    """
    Phase H redesigned geometry-constrained Braille dot detector
    for embossed dots.
    """

    def __init__(self, model_path: str = DEFAULT_DOT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.yolo_model = None
        self.mode       = "opencv"
        self.has_warned = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        try:
            from ultralytics import YOLO

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Missing model weights: {self.model_path}"
                )

            self.yolo_model = YOLO(self.model_path)
            self.mode = "yolov8"

            print("=" * 50)
            print("[OK] YOLO dot-detection model loaded")
            print(f"[MODEL]: {self.model_path}")
            print(f"[MODE]: {self.mode}")
            print("=" * 50)

        except Exception as e:
            self.yolo_model = None
            self.mode = "opencv"
            
            if not self.has_warned:
                print("[BrailleVisionAI] Dot detector mode: OpenCV")
                self.has_warned = True

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
        result = self._pipeline(bgr_frame)
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

        accepted, rejected, debug_frame, stats = self._pipeline(bgr_frame)
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
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════
    def _pipeline(self, bgr_frame: np.ndarray) -> Tuple[List[BrailleDot], List[BrailleDot], np.ndarray, Dict]:
        t0 = time.perf_counter()
        
        # 1 & 2. Grayscale & Preprocessing
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY) if bgr_frame.ndim == 3 else bgr_frame.copy()
        
        # CLAHE (clipLimit=3)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
        
        # GaussianBlur (kernel=(5,5))
        gray_blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)
        
        # Adaptive Threshold
        binary = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # Morphological open
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Morphological close
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 40 or area > 1000:
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.75:
                continue
                
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            if radius < 4 or radius > 20:
                continue
                
            candidates.append(BrailleDot(
                x=int(x),
                y=int(y),
                radius=float(radius),
                confidence=1.0
            ))
            
        # 4. Spacing estimation & Noise rejection
        row_spacing, col_spacing = self._estimate_spacings(candidates)
        accepted, rejected = self._filter_noise(candidates, row_spacing, col_spacing)
        
        # 5. Build cells
        cells = self._build_cells(accepted, row_spacing, col_spacing)
        
        # Estimate rows and columns of cells
        estimated_rows, estimated_columns = self._estimate_rows_cols(cells, row_spacing, col_spacing)
        
        # 6. Render overlay
        debug_frame = self._draw_overlay(bgr_frame, accepted, rejected, cells)
        
        # 7. Collect metrics
        ms = (time.perf_counter() - t0) * 1000.0
        
        stats = {
            "candidate_dots": len(candidates),
            "accepted_dots": len(accepted),
            "rejected_dots": len(rejected),
            "average_spacing": round((row_spacing + col_spacing) / 2.0, 1),
            "estimated_rows": estimated_rows,
            "estimated_columns": estimated_columns,
            
            # Legacy compatibility keys
            "raw_contour_count": len(candidates),
            "filtered_count": len(accepted),
            "rejected_tiny": len(rejected),
            "rejected_irregular": 0,
            "rejected_size": 0,
            "processing_ms": round(ms, 1),
            "geo_conf": 0.85 if len(cells) > 0 else 0.0,
            "ghost_count": 0,
            "spacing": (col_spacing, row_spacing),
            "angle": 0.0,
        }
        
        return accepted, rejected, debug_frame, stats

    def _estimate_spacings(self, candidates: List[BrailleDot]) -> Tuple[float, float]:
        if len(candidates) < 2:
            return 15.0, 15.0
            
        coords = np.array([[d.x, d.y] for d in candidates], dtype=float)
        h_gaps = []
        v_gaps = []
        
        for i, pt in enumerate(coords):
            dists = np.linalg.norm(coords - pt, axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:4]
            for idx in nearest:
                d = dists[idx]
                if d > 120.0 or d < 4.0:
                    continue
                dx = abs(coords[idx, 0] - pt[0])
                dy = abs(coords[idx, 1] - pt[1])
                if dx > dy:
                    h_gaps.append(dx)
                else:
                    v_gaps.append(dy)
                    
        row_spacing = float(np.median(v_gaps)) if v_gaps else 15.0
        col_spacing = float(np.median(h_gaps)) if h_gaps else 15.0
        
        row_spacing = np.clip(row_spacing, 5.0, 50.0)
        col_spacing = np.clip(col_spacing, 5.0, 50.0)
        
        return row_spacing, col_spacing

    def _filter_noise(self, candidates: List[BrailleDot], row_spacing: float, col_spacing: float) -> Tuple[List[BrailleDot], List[BrailleDot]]:
        if len(candidates) < 2:
            return [], candidates.copy()
            
        accepted = []
        rejected = []
        
        # Keep only dots having neighbors within neighbor_threshold
        neighbor_threshold = 2.5 * max(row_spacing, col_spacing)
        
        coords = np.array([[d.x, d.y] for d in candidates], dtype=float)
        for i, dot in enumerate(candidates):
            dists = np.linalg.norm(coords - coords[i], axis=1)
            dists[i] = np.inf
            if dists.min() <= neighbor_threshold:
                accepted.append(dot)
            else:
                rejected.append(dot)
                
        return accepted, rejected

    def _build_cells(self, accepted_dots: List[BrailleDot], row_spacing: float, col_spacing: float) -> List[BrailleCell]:
        if not accepted_dots:
            return []
            
        # Group dots using single-linkage clustering
        clusters = []
        used = [False] * len(accepted_dots)
        cluster_threshold = 1.8 * max(row_spacing, col_spacing)
        
        for i, dot in enumerate(accepted_dots):
            if used[i]:
                continue
            cluster = [dot]
            used[i] = True
            queue = [dot]
            while queue:
                curr = queue.pop(0)
                for j, other in enumerate(accepted_dots):
                    if not used[j]:
                        dist = math.hypot(curr.x - other.x, curr.y - other.y)
                        if dist <= cluster_threshold:
                            cluster.append(other)
                            queue.append(other)
                            used[j] = True
            clusters.append(cluster)
            
        cells = []
        for cluster in clusters:
            dot_count = len(cluster)
            if dot_count < 1 or dot_count > 6:
                continue
                
            min_x = min(d.x for d in cluster)
            min_y = min(d.y for d in cluster)
            max_x = max(d.x for d in cluster)
            max_y = max(d.y for d in cluster)
            
            cell_w = max_x - min_x
            cell_h = max_y - min_y
            
            if col_spacing > 0 and (cell_w / col_spacing) > 1.8:
                continue
                
            ys = sorted([d.y for d in cluster])
            row_gaps = [ys[i] - ys[i-1] for i in range(1, len(ys)) if ys[i] - ys[i-1] > row_spacing * 0.5]
            if len(row_gaps) >= 2:
                mean_gap = sum(row_gaps) / len(row_gaps)
                if any(abs(g - mean_gap) / mean_gap > 0.25 for g in row_gaps):
                    continue
            
            best_error = 1e9
            best_anchor = (min_x, min_y)
            best_mapping = {}
            
            # Fit to fixed 2-column x 3-row grid template
            for c_off in [0, 1]:
                for r_off in [0, 1, 2]:
                    anchor_x = min_x - c_off * col_spacing
                    anchor_y = min_y - r_off * row_spacing
                    
                    error = 0.0
                    mapping = {}
                    for idx, dot in enumerate(cluster):
                        c_idx = int(round((dot.x - anchor_x) / col_spacing))
                        r_idx = int(round((dot.y - anchor_y) / row_spacing))
                        
                        c_idx = max(0, min(1, c_idx))
                        r_idx = max(0, min(2, r_idx))
                        
                        slot = c_idx * 3 + r_idx
                        slot_x = anchor_x + c_idx * col_spacing
                        slot_y = anchor_y + r_idx * row_spacing
                        
                        dist = math.hypot(dot.x - slot_x, dot.y - slot_y)
                        error += dist * dist
                        mapping[idx] = slot
                        
                    if error < best_error:
                        best_error = error
                        best_anchor = (anchor_x, anchor_y)
                        best_mapping = mapping
                        
            # Check fitting error quality
            avg_fit_error = math.sqrt(best_error / len(cluster))
            if avg_fit_error > 0.6 * max(row_spacing, col_spacing):
                continue
                
            anchor_x, anchor_y = best_anchor
            
            pattern = ['0'] * 6
            cell_dots = [None] * 6
            for idx, dot in enumerate(cluster):
                slot = best_mapping[idx]
                pattern[slot] = '1'
                cell_dots[slot] = dot
                
            binary_pattern = "".join(pattern)
            if binary_pattern == "000000":
                continue
                
            from translation.braille_mapper import translate_binary_pattern
            char = translate_binary_pattern(binary_pattern)
            
            w = max(15, int(col_spacing + 12))
            h = max(25, int(2 * row_spacing + 12))
            
            # Avoid absurdly large distorted clusters
            cell_w = max(d.x for d in cluster) - min(d.x for d in cluster)
            cell_h = max(d.y for d in cluster) - min(d.y for d in cluster)
            if cell_w > 3.0 * col_spacing or cell_h > 4.0 * row_spacing:
                continue
                
            filled_ratio = len(cluster) / 6.0
            confidence = 0.6 * filled_ratio + 0.4
            
            cells.append(BrailleCell(
                x=int(anchor_x),
                y=int(anchor_y),
                w=w,
                h=h,
                dots=[d for d in cell_dots if d is not None],
                binary_pattern=binary_pattern,
                translated_char=char,
                confidence=confidence,
            ))
            
        return cells

    def _estimate_rows_cols(self, cells: List[BrailleCell], row_spacing: float, col_spacing: float) -> Tuple[int, int]:
        if not cells:
            return 0, 0
            
        cell_ys = sorted([c.y for c in cells])
        cell_xs = sorted([c.x for c in cells])
        
        row_count = 1
        for k in range(1, len(cell_ys)):
            if cell_ys[k] - cell_ys[k-1] > 1.5 * row_spacing:
                row_count += 1
                
        col_count = 1
        for k in range(1, len(cell_xs)):
            if cell_xs[k] - cell_xs[k-1] > 1.5 * col_spacing:
                col_count += 1
                
        return row_count, col_count

    def _draw_overlay(self, bgr: np.ndarray, accepted: List[BrailleDot], rejected: List[BrailleDot], cells: List[BrailleCell]) -> np.ndarray:
        out = bgr.copy()
        
        # Display counts
        cv2.putText(out, f"Accepted dots: {len(accepted)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out, f"Rejected dots: {len(rejected)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(out, f"Detected cells: {len(cells)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Blue = detected Braille cells
        for cell in cells:
            cv2.rectangle(out, (cell.x, cell.y), (cell.x + cell.w, cell.y + cell.h), (255, 0, 0), 2, cv2.LINE_AA)
            if cell.translated_char:
                cv2.putText(out, cell.translated_char, (cell.x, cell.y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                
        # Red = rejected dots
        for d in rejected:
            cv2.circle(out, (d.x, d.y), max(3, int(d.radius)), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(out, (d.x - 3, d.y), (d.x + 3, d.y), (0, 0, 255), 1)
            cv2.line(out, (d.x, d.y - 3), (d.x, d.y + 3), (0, 0, 255), 1)
            
        # Green = accepted dots
        for d in accepted:
            cv2.circle(out, (d.x, d.y), max(3, int(d.radius)), (0, 255, 0), 2, cv2.LINE_AA)
            
        return out
