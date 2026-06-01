"""Quick validation script for refactored Braille detection modules."""
import sys
sys.path.insert(0, '.')

results = []

# 1. dot_detector
try:
    from detection.dot_detector import BrailleDotDetector, _local_contrast
    import numpy as np
    det = BrailleDotDetector()
    
    # Test local contrast computation (replaces emboss score in H.5)
    test_patch = np.random.randint(100, 200, (20, 20), dtype=np.uint8)
    score = _local_contrast(test_patch, 10, 10, 5)
    assert score >= 0.0
    
    # Test pipeline with a dummy image
    dummy = np.ones((200, 300, 3), dtype=np.uint8) * 200
    accepted, rejected, debug, stats = det.detect_with_debug(
        dummy, avg_spacing=15.0, detect_mode='balanced', demo_mode=False
    )
    results.append('[OK] dot_detector: accepted=' + str(len(accepted)) + ' rejected=' + str(len(rejected)))
except Exception as e:
    results.append('[FAIL] dot_detector: ' + str(e))


# 2. cell_extractor
try:
    from detection.cell_extractor import BrailleCellExtractor, confidence_tier
    ext = BrailleCellExtractor(demo_mode=True)
    assert ext.demo_mode == True
    assert ext.MIN_CELL_CONF == 0.30
    
    # Tier tests
    t1 = confidence_tier(0.25)
    t2 = confidence_tier(0.50)
    t3 = confidence_tier(0.70)
    t4 = confidence_tier(0.90)
    results.append('[OK] cell_extractor: tiers=' + str([t1, t2, t3, t4]))
except Exception as e:
    results.append('[FAIL] cell_extractor: ' + str(e))

# 3. inference module
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location('inference', 'detection/inference.py')
    # Just parse it
    with open('detection/inference.py') as f:
        code = f.read()
    assert 'demo_mode' in code
    assert 'debug_mode' in code
    results.append('[OK] inference.py: demo_mode and debug_mode present')
except Exception as e:
    results.append('[FAIL] inference: ' + str(e))

# 4. calibration_profiles
try:
    from detection.calibration_profiles import get_profile
    p = get_profile('DEMO_OPTIMIZED')
    assert p.get('demo_mode') == True, 'demo_mode not True'
    assert p.get('detect_mode') == 'strict', 'detect_mode not strict'
    results.append('[OK] calibration_profiles: DEMO_OPTIMIZED detect_mode=strict demo_mode=True')
except Exception as e:
    results.append('[FAIL] calibration_profiles: ' + str(e))

# 5. geometry_utils key functions
try:
    from detection.geometry_utils import (
        estimate_braille_spacings, cluster_dots_into_cells,
        validate_cell_geometry, fit_cluster_to_grid,
        score_cluster_as_braille_cell, detect_braille_row_structure
    )
    results.append('[OK] geometry_utils: all key functions importable')
except Exception as e:
    results.append('[FAIL] geometry_utils: ' + str(e))

# 6. Test suite
try:
    import os
    assert os.path.exists('tests/demo_validation/evaluate_demo_accuracy.py')
    results.append('[OK] tests/demo_validation/evaluate_demo_accuracy.py exists')
except Exception as e:
    results.append('[FAIL] test suite: ' + str(e))

print()
print('='*60)
print('  BrailleVisionAI Refactor Validation')
print('='*60)
for r in results:
    print(' ', r)
print()
ok_count = sum(1 for r in results if r.startswith('[OK]'))
fail_count = len(results) - ok_count
print('  PASSED:', ok_count, '/', len(results))
if fail_count == 0:
    print('  All validation checks passed!')
else:
    print('  FAILED:', fail_count, 'check(s) need attention.')
print()
