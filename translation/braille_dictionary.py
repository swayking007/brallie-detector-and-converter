"""
============================================================
BrailleVisionAI — Phase F  |  Braille Dictionary
translation/braille_dictionary.py
============================================================

PURPOSE
-------
The single source of truth for ALL Braille-to-English pattern
mappings used by the Phase F translation engine.

Braille Cell Dot Layout (6 dots):
    Dot 1  Dot 4
    Dot 2  Dot 5
    Dot 3  Dot 6

Binary pattern = 6-char string, one digit per dot in order:
    "100000" → dot 1 raised only → 'a'
    "110000" → dots 1+2 raised   → 'b'

Special prefix cells (indicators):
    CAPITAL_INDICATOR  = "000001"  → next char is uppercase
    NUMBER_INDICATOR   = "010111"  → next chars are digits until space

============================================================
"""

# ── Braille Alphabet (Grade 1 — lowercase letters) ─────────
LETTERS: dict[str, str] = {
    "100000": "a",
    "110000": "b",
    "100100": "c",
    "100110": "d",
    "100010": "e",
    "110100": "f",
    "110110": "g",
    "110010": "h",
    "010100": "i",
    "010110": "j",
    "101000": "k",
    "111000": "l",
    "101100": "m",
    "101110": "n",
    "101010": "o",
    "111100": "p",
    "111110": "q",
    "111010": "r",
    "011100": "s",
    "011110": "t",
    "101001": "u",
    "111001": "v",
    "010111": "w",
    "101101": "x",
    "101111": "y",
    "101011": "z",
}

# ── Number mode: same patterns as a–j but used after number indicator ──
# In Braille, numbers use the same dot patterns as a–j
NUMBERS: dict[str, str] = {
    "100000": "1",   # a pattern → 1
    "110000": "2",   # b pattern → 2
    "100100": "3",   # c pattern → 3
    "100110": "4",   # d pattern → 4
    "100010": "5",   # e pattern → 5
    "110100": "6",   # f pattern → 6
    "110110": "7",   # g pattern → 7
    "110010": "8",   # h pattern → 8
    "010100": "9",   # i pattern → 9
    "010110": "0",   # j pattern → 0
}

# ── Punctuation & Special Symbols ──────────────────────────
PUNCTUATION: dict[str, str] = {
    "000000": " ",   # empty cell → space
    "010000": ",",   # comma
    "011000": ";",   # semicolon
    "010010": ":",   # colon
    "010001": ".",   # period / full stop
    "011001": "!",   # exclamation mark
    "001101": "?",   # question mark
    "001100": "\"",  # open/close quotation mark
    "001000": "'",   # apostrophe
    "001001": "-",   # hyphen / dash
    "110001": "(",   # open parenthesis
    "001110": ")",   # close parenthesis
    "000100": "@",   # at sign (used in some notations)
}

# ── Special Indicator Cells ─────────────────────────────────
# These cells modify the meaning of the NEXT cell(s).
CAPITAL_INDICATOR = "000001"   # Next character should be uppercase
NUMBER_INDICATOR  = "010111"   # Switch into number mode until space

# ── Master dictionary (letters + punctuation, not numbers — handled by mode) ──
MASTER_DICT: dict[str, str] = {}
MASTER_DICT.update(LETTERS)
MASTER_DICT.update(PUNCTUATION)

# ── Human-friendly display name for indicator cells ────────
INDICATOR_LABELS: dict[str, str] = {
    CAPITAL_INDICATOR: "[CAP]",
    NUMBER_INDICATOR:  "[NUM]",
}

def get_char(pattern: str, number_mode: bool = False) -> str:
    """
    Look up a single 6-bit Braille pattern.

    Args:
        pattern:     6-character binary string, e.g. "100000".
        number_mode: If True, prefer the digit mapping.

    Returns:
        The corresponding English character, or '?' if unknown.
    """
    if not pattern or len(pattern) != 6:
        return "?"

    # Indicator cells are not characters themselves
    if pattern == CAPITAL_INDICATOR:
        return ""   # consumed silently; caller handles capitalisation
    if pattern == NUMBER_INDICATOR:
        return ""   # consumed silently; caller enables number_mode

    if number_mode:
        char = NUMBERS.get(pattern)
        if char is not None:
            return char

    return MASTER_DICT.get(pattern, "?")


def is_known(pattern: str) -> bool:
    """Return True if this pattern is in the master dictionary or is an indicator."""
    return (
        pattern in MASTER_DICT
        or pattern == CAPITAL_INDICATOR
        or pattern == NUMBER_INDICATOR
    )
