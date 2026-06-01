"""Quick recall test: draw 30 dots across 5 Braille cells, check detection."""
import cv2
import numpy as np
from detection.dot_detector import BrailleDotDetector

img = np.ones((300, 400, 3), dtype=np.uint8) * 195
sp = 15

# Draw 5 full Braille cells (2 cols x 3 rows each = 30 dots)
for cell in range(5):
    for c in range(2):
        for r in range(3):
            cx = 80 + c * sp + cell * 40
            cy = 60 + r * sp
            cv2.circle(img, (cx, cy), 6, (75, 75, 75), -1)

det = BrailleDotDetector()
a, r, d, s = det.detect_with_debug(img, avg_spacing=15.0)

print("Drew 30 dots")
print("Accepted:", len(a))
print("Rejected:", len(r))
print("Geo conf:", s["geo_conf"])
print("Ghosts:", s["ghost_count"])
print("Processing:", s["processing_ms"], "ms")

if len(a) >= 20:
    print("RECALL TEST PASSED - detected", len(a), "of 30 dots")
else:
    print("RECALL TEST WARNING - only", len(a), "of 30 dots detected")
