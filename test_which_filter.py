import cv2
import numpy as np
import os

img_path = r"c:\Users\User\Desktop\braille\tests\demo_validation\hello.png"
bgr = cv2.imread(img_path)
gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)
gray_blur = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

binary = cv2.adaptiveThreshold(
    gray_blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

filters = [
    ("Area (20-500)", {"filterByArea": True, "minArea": 20, "maxArea": 500}),
    ("Circularity (0.5)", {"filterByCircularity": True, "minCircularity": 0.5}),
    ("Convexity (0.5)", {"filterByConvexity": True, "minConvexity": 0.5}),
    ("Inertia (0.2)", {"filterByInertia": True, "minInertiaRatio": 0.2}),
]

for name, kwargs in filters:
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 5
    params.maxThreshold = 255
    params.thresholdStep = 5
    params.filterByColor = True
    params.blobColor = 255
    
    # Enable only this filter
    params.filterByArea = kwargs.get("filterByArea", False)
    if params.filterByArea:
        params.minArea = kwargs["minArea"]
        params.maxArea = kwargs["maxArea"]
        
    params.filterByCircularity = kwargs.get("filterByCircularity", False)
    if params.filterByCircularity:
        params.minCircularity = kwargs["minCircularity"]
        
    params.filterByConvexity = kwargs.get("filterByConvexity", False)
    if params.filterByConvexity:
        params.minConvexity = kwargs["minConvexity"]
        
    params.filterByInertia = kwargs.get("filterByInertia", False)
    if params.filterByInertia:
        params.minInertiaRatio = kwargs["minInertiaRatio"]
        
    detector = cv2.SimpleBlobDetector_create(params)
    kp = detector.detect(opened)
    print(f"Filter '{name}': detected {len(kp)} blobs")
