"""
============================================================
BrailleVisionAI — Phase E  |  Braille Pattern Data Models
detection/braille_pattern.py
============================================================

PURPOSE
-------
Defines dataclasses for Braille dots and Braille cells,
supporting structured storage of 6-dot patterns.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class BrailleDot:
    """
    Represents an individual detected Braille dot.
    """
    x: int
    y: int
    radius: float
    confidence: float = 1.0

@dataclass
class BrailleCell:
    """
    Represents a single grouped Braille character cell.
    
    A standard cell is composed of up to 6 dots:
    (1) (4)
    (2) (5)
    (3) (6)
    """
    x: int  # bounding box top-left x
    y: int  # bounding box top-left y
    w: int  # bounding box width
    h: int  # bounding box height
    dots: List[BrailleDot] = field(default_factory=list)
    binary_pattern: str = "000000"  # 6-digit binary string, e.g., '100000'
    translated_char: str = "?"      # Translated character from braille_mapper
    confidence: float = 1.0
    row_idx: int = -1               # Row index in reading layout
    col_idx: int = -1               # Column index in reading layout
