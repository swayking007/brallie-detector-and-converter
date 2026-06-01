import cv2
import os
from detection.dot_detector import BrailleDotDetector

img_path = os.path.join('c:/Users/User/Desktop/braille', 'tests', 'demo_validation', 'hello.png')
# ensure path exists
if not os.path.exists(img_path):
    print('Image not found')
    exit(1)

bgr = cv2.imread(img_path)
det = BrailleDotDetector()
accepted, rejected, debug, stats = det.detect_with_debug(bgr, avg_spacing=20)
print('Accepted dots:', len(accepted))
print('Rejected dots:', len(rejected))
print('Stats:', stats)
# Save debug overlay for inspection
cv2.imwrite('c:/Users/User/Desktop/braille/debug_hello.png', debug)
print('Debug image saved to debug_hello.png')
