"""Quick integration test for Phase H.5 geometry-constrained pipeline."""
import cv2
import numpy as np
from detection.dot_detector import BrailleDotDetector
from detection.grid_engine import BrailleGridEngine, estimate_spacing

# Create synthetic image with embossed dot pattern
img = np.ones((200, 300, 3), dtype=np.uint8) * 200

# Draw a 2x3 Braille cell at (100, 50)
h_sp, v_sp = 15, 15
for col in range(2):
    for row in range(3):
        cx = 100 + col * h_sp
        cy = 50 + row * v_sp
        cv2.circle(img, (cx, cy), 6, (80, 80, 80), -1)

# Draw a second cell
for col in range(2):
    for row in [0, 2]:
        cx = 160 + col * h_sp
        cy = 50 + row * v_sp
        cv2.circle(img, (cx, cy), 6, (80, 80, 80), -1)

det = BrailleDotDetector()
accepted, rejected, debug_frame, stats = det.detect_with_debug(img, avg_spacing=15.0)

print(f"Accepted dots: {len(accepted)}")
print(f"Rejected dots: {len(rejected)}")
print(f"Geometry confidence: {stats['geo_conf']}")
print(f"Ghost count: {stats['ghost_count']}")
print(f"Processing time: {stats['processing_ms']}ms")
print(f"Raw contours: {stats['raw_contour_count']}")

# Also test the full inference pipeline import
from detection.inference import run_cell_extraction
dots, cells, annotated, tr = run_cell_extraction(img, avg_spacing=15.0)
print(f"Cells extracted: {len(cells)}")
print("Pipeline test PASSED")
