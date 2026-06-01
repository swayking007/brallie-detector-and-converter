"""
============================================================
BrailleVisionAI — Demo Validation Test Suite
tests/demo_validation/evaluate_demo_accuracy.py
============================================================

Evaluates detection accuracy on synthetic Braille test images.

Metrics:
  - detected cells
  - correctly translated chars
  - false positives
  - processing time
  - confidence

Run:
    python -m tests.demo_validation.evaluate_demo_accuracy

Or from project root:
    python tests/demo_validation/evaluate_demo_accuracy.py
"""

from __future__ import annotations

import sys
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# ── Ensure project root is on path ───────────────────────────
_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    print("[ERROR] OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)

# ── Braille character -> 6-bit binary pattern ────────────────
# Standard Grade 1 Braille (dot positions 1-6: columns L/R, rows T/M/B)
BRAILLE_PATTERNS: Dict[str, str] = {
    'a': '100000',
    'b': '110000',
    'c': '100100',
    'd': '100110',
    'e': '100010',
    'f': '110100',
    'g': '110110',
    'h': '110010',
    'i': '010100',
    'j': '010110',
    'k': '101000',
    'l': '111000',
    'm': '101100',
    'n': '101110',
    'o': '101010',
    'p': '111100',
    'q': '111110',
    'r': '111010',
    's': '011100',
    't': '011110',
    'u': '101001',
    'v': '111001',
    'w': '010111',
    'x': '101101',
    'y': '101111',
    'z': '101011',
    ' ': '000000',
}

# ── Test cases ───────────────────────────────────────────────
TEST_CASES: List[Dict] = [
    {"name": "hello",        "word": "hello",        "filename": "hello.png"},
    {"name": "abc",          "word": "abc",          "filename": "abc.png"},
    {"name": "thank_you",    "word": "thankyou",     "filename": "thank_you.png"},
    {"name": "good_morning", "word": "goodmorning",  "filename": "good_morning.png"},
    {"name": "hi",           "word": "hi",           "filename": "hi.png"},
]

# ── Image generation ──────────────────────────────────────────

def _braille_dot_positions(
    pattern: str, cell_x: int, cell_y: int,
    dot_spacing: int = 18, cell_width: int = 28,
) -> List[Tuple[int, int]]:
    """Return (x,y) pixel positions for filled dots in a Braille pattern."""
    # Layout: 2 cols (left=0, right=1) × 3 rows (top=0, mid=1, bot=2)
    col_xs = [cell_x + dot_spacing // 2,
               cell_x + dot_spacing // 2 + dot_spacing]
    row_ys = [cell_y + dot_spacing // 2,
               cell_y + dot_spacing // 2 + dot_spacing,
               cell_y + dot_spacing // 2 + dot_spacing * 2]
    positions = []
    slots = [
        (0, 0), (0, 1), (0, 2),   # left col: pos 1,2,3
        (1, 0), (1, 1), (1, 2),   # right col: pos 4,5,6
    ]
    for i, (col, row) in enumerate(slots):
        if i < len(pattern) and pattern[i] == '1':
            positions.append((col_xs[col], row_ys[row]))
    return positions


def generate_braille_image(
    word: str,
    dot_radius: int = 6,
    dot_spacing: int = 20,
    cell_gap: int = 8,
    margin: int = 30,
    bg_brightness: int = 230,
    dot_brightness: int = 140,
    add_noise: bool = False,
    side_light: bool = True,
) -> np.ndarray:
    """
    Generate a synthetic embossed Braille image for testing.

    Parameters
    ----------
    word           : text to render (characters mapped to Braille patterns)
    dot_radius     : radius of each dot in pixels
    dot_spacing    : vertical spacing between dots within a cell
    cell_gap       : horizontal gap between cells
    margin         : image border margin
    bg_brightness  : background grey value (0-255)
    dot_brightness : dot grey value (0-255, < bg = darker dot)
    add_noise      : add Gaussian noise to simulate real paper
    side_light     : add emboss-style highlight/shadow simulation
    """
    # Remove spaces from word for layout calculation
    chars = [c for c in word.lower() if c in BRAILLE_PATTERNS]
    if not chars:
        chars = ['a']   # fallback

    cell_width  = dot_spacing + dot_spacing + 4
    cell_height = dot_spacing * 3 + 4

    # Image dimensions
    total_w = margin * 2 + len(chars) * (cell_width + cell_gap) - cell_gap
    total_h = margin * 2 + cell_height

    # Create background
    img = np.full((total_h, total_w), bg_brightness, dtype=np.uint8)

    # Draw each character
    for i, char in enumerate(chars):
        pattern = BRAILLE_PATTERNS.get(char, '000000')
        cell_x  = margin + i * (cell_width + cell_gap)
        cell_y  = margin

        positions = _braille_dot_positions(
            pattern, cell_x, cell_y, dot_spacing
        )

        for (dx, dy) in positions:
            # Draw dark dot (embossed dot appears darker under side lighting)
            cv2.circle(img, (dx, dy), dot_radius, dot_brightness, -1, cv2.LINE_AA)

            if side_light:
                # Add highlight arc (top-left lighter)
                hl_offset = max(1, dot_radius // 3)
                hl_x = dx - hl_offset
                hl_y = dy - hl_offset
                cv2.circle(img, (hl_x, hl_y), max(1, dot_radius // 3),
                           min(255, bg_brightness + 20), -1, cv2.LINE_AA)
                # Add shadow arc (bottom-right darker)
                sh_x = dx + hl_offset
                sh_y = dy + hl_offset
                cv2.circle(img, (sh_x, sh_y), max(1, dot_radius // 3),
                           max(0, dot_brightness - 20), -1, cv2.LINE_AA)

    # Add Gaussian noise
    if add_noise:
        noise = np.random.normal(0, 8, img.shape).astype(np.float32)
        img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def save_test_images(output_dir: Path) -> None:
    """Generate and save all test images to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for tc in TEST_CASES:
        img = generate_braille_image(tc["word"], add_noise=False, side_light=True)
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        path = output_dir / tc["filename"]
        cv2.imwrite(str(path), bgr)
        print(f"  [SAVED] {path.name}  ({img.shape[1]}×{img.shape[0]}px)")


# ── Evaluation ────────────────────────────────────────────────

def evaluate_image(
    bgr: np.ndarray,
    expected_text: str,
    demo_mode: bool = True,
) -> Dict:
    """
    Run full pipeline on a BGR image and return accuracy metrics.
    """
    from detection.dot_detector  import BrailleDotDetector
    from detection.cell_extractor import BrailleCellExtractor
    from detection.geometry_utils import (
        cluster_dots_into_cells, validate_cell_geometry, geometry_confidence
    )

    t0     = time.perf_counter()
    det    = BrailleDotDetector()
    ext    = BrailleCellExtractor(demo_mode=demo_mode)

    # Estimate spacing from image width (synthetic images use fixed spacing)
    avg_sp = max(10.0, bgr.shape[1] * 0.015)

    accepted, rejected, _, stats = det.detect_with_debug(
        bgr, avg_spacing=avg_sp,
        detect_mode="strict" if demo_mode else "balanced",
        demo_mode=demo_mode,
    )

    cells = ext.extract_cells(accepted, avg_sp)
    ms    = (time.perf_counter() - t0) * 1000

    # Compute accuracy
    detected_text = "".join(c.translated_char for c in cells
                            if c.translated_char and c.translated_char != '?')
    expected_chars = [c for c in expected_text.lower() if c.strip()]
    detected_chars = [c for c in detected_text.lower() if c.strip()]

    # Character accuracy
    correct = sum(1 for a, b in zip(expected_chars, detected_chars) if a == b)
    total   = max(1, len(expected_chars))
    char_acc = correct / total

    # False positive count (extra cells beyond expected)
    false_pos = max(0, len(cells) - len(expected_chars))

    avg_conf = float(np.mean([c.confidence for c in cells])) if cells else 0.0

    return {
        "detected_cells":   len(cells),
        "expected_cells":   len(expected_chars),
        "correct_chars":    correct,
        "char_accuracy":    char_acc,
        "false_positives":  false_pos,
        "raw_candidates":   len(accepted),
        "rejected_dots":    len(rejected),
        "avg_confidence":   avg_conf,
        "processing_ms":    ms,
        "detected_text":    detected_text,
        "expected_text":    expected_text,
    }


def print_summary_table(results: List[Dict]) -> None:
    """Print a formatted accuracy summary table."""
    print("\n" + "="*90)
    print("  BrailleVisionAI — Demo Validation Summary")
    print("="*90)
    hdr = (
        f"{'Test':<18} {'Expected':<14} {'Detected':<14} "
        f"{'Cells':>6} {'Acc':>6} {'FP':>4} "
        f"{'Conf':>6} {'ms':>7}"
    )
    print(hdr)
    print("-"*90)

    total_correct  = 0
    total_expected = 0
    total_ms       = 0.0
    total_fp       = 0

    for r in results:
        acc_pct  = f"{r['char_accuracy']:.0%}"
        conf_pct = f"{r['avg_confidence']:.0%}"
        ms_str   = f"{r['processing_ms']:.0f}ms"
        det_short = r['detected_text'][:12] if r['detected_text'] else '—'
        exp_short = r['expected_text'][:12] if r['expected_text'] else '—'

        print(
            f"  {r['name']:<16} {exp_short:<14} {det_short:<14} "
            f"{r['detected_cells']:>6} {acc_pct:>6} {r['false_positives']:>4} "
            f"{conf_pct:>6} {ms_str:>7}"
        )

        total_correct  += r['correct_chars']
        total_expected += r['expected_cells']
        total_ms       += r['processing_ms']
        total_fp       += r['false_positives']

    print("-"*90)
    overall_acc = total_correct / max(1, total_expected)
    avg_ms      = total_ms / max(1, len(results))
    print(
        f"  {'OVERALL':<16} {'':<14} {'':<14} "
        f"{total_correct:>6} {overall_acc:.0%} {total_fp:>4} "
        f"{'':>6} {avg_ms:.0f}ms avg"
    )
    print("="*90)
    print()

    # Performance gate
    if avg_ms < 1000:
        print(f"  ✅ Speed target MET: avg {avg_ms:.0f}ms < 1000ms")
    else:
        print(f"  ⚠️  Speed target MISSED: avg {avg_ms:.0f}ms > 1000ms")

    if total_fp == 0:
        print(f"  ✅ False positives: ZERO")
    else:
        print(f"  ⚠️  False positives: {total_fp} total")

    if overall_acc >= 0.70:
        print(f"  ✅ Accuracy: {overall_acc:.0%} (target: ≥70%)")
    else:
        print(f"  ⚠️  Accuracy: {overall_acc:.0%} (target: ≥70%)")
    print()


# ── Entry point ───────────────────────────────────────────────

def main():
    print("\n[BrailleVisionAI] Demo Validation Test Suite")
    print("─" * 60)

    # Paths
    test_dir   = Path(__file__).parent
    images_dir = test_dir

    # Generate test images if not present
    any_missing = any(not (images_dir / tc["filename"]).exists() for tc in TEST_CASES)
    if any_missing:
        print("\n[INFO] Generating synthetic test images…")
        save_test_images(images_dir)

    # Run evaluation
    print("\n[INFO] Running evaluation pipeline…")
    results = []
    for tc in TEST_CASES:
        img_path = images_dir / tc["filename"]
        if not img_path.exists():
            print(f"  [SKIP] {tc['filename']} not found")
            continue

        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [SKIP] Could not load {tc['filename']}")
            continue

        print(f"  Evaluating: {tc['name']} ({tc['word']})…", end="", flush=True)
        result = evaluate_image(bgr, tc["word"], demo_mode=True)
        result["name"] = tc["name"]
        results.append(result)
        print(f" {result['char_accuracy']:.0%} accuracy, {result['processing_ms']:.0f}ms")

    if results:
        print_summary_table(results)
    else:
        print("[ERROR] No results generated.")


if __name__ == "__main__":
    main()
