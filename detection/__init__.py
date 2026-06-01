# BrailleVisionAI — detection package  |  Phase D, E & F
from detection.braille_detector  import BraillePresenceDetector, DetectionResult
from detection.dot_detector      import BrailleDotDetector
from detection.cell_extractor    import BrailleCellExtractor
from detection.braille_pattern   import BrailleDot, BrailleCell
from detection.overlay_renderer  import draw_braille_overlays
from detection.inference import (
    get_detector,
    get_dot_detector,
    get_cell_extractor,
    get_translator,
    get_text_builder,
    run_detection,
    run_cell_extraction,
    TRANSLATION_OK,
)

__all__ = [
    "BraillePresenceDetector",
    "DetectionResult",
    "BrailleDotDetector",
    "BrailleCellExtractor",
    "BrailleDot",
    "BrailleCell",
    "draw_braille_overlays",
    "get_detector",
    "get_dot_detector",
    "get_cell_extractor",
    "get_translator",
    "get_text_builder",
    "run_detection",
    "run_cell_extraction",
    "TRANSLATION_OK",
]
