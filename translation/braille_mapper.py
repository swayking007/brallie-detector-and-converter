"""
============================================================
BrailleVisionAI — Phase E  |  Braille Mapping Logic
translation/braille_mapper.py
============================================================

PURPOSE
-------
Provides the mapping from binary Braille dot patterns to
English characters.

Braille Cell Dot Numbering:
(1) (4)
(2) (5)
(3) (6)

A binary pattern is a 6-character string where 1 means raised 
and 0 means absent. For example, '100000' is 'a'.

============================================================
"""

# Standard English Grade 1 Braille mapping
BRAILLE_TO_ENGLISH = {
    "100000": "a", "110000": "b", "100100": "c", "100110": "d", "100010": "e",
    "110100": "f", "110110": "g", "110010": "h", "010100": "i", "010110": "j",
    "101000": "k", "111000": "l", "101100": "m", "101110": "n", "101010": "o",
    "111100": "p", "111110": "q", "111010": "r", "011100": "s", "011110": "t",
    "101001": "u", "111001": "v", "010111": "w", "101101": "x", "101111": "y",
    "101011": "z",
    "000000": " ",
}

def translate_binary_pattern(pattern: str) -> str:
    """
    Convert a 6-character binary string (e.g., '100000') into an English character.
    Returns '?' if the pattern is not recognized.
    """
    return BRAILLE_TO_ENGLISH.get(pattern, "?")
