# BrailleVisionAI — translation package  |  Phase F
from translation.braille_dictionary   import (
    LETTERS, NUMBERS, PUNCTUATION, MASTER_DICT,
    CAPITAL_INDICATOR, NUMBER_INDICATOR,
    get_char, is_known,
)
from translation.translator_engine    import TranslatorEngine, TranslationResult, CharResult
from translation.confidence_handler   import (
    tier_for, analyse,
    ConfidenceTier, ConfidenceSummary,
    COLOUR_HIGH_HEX, COLOUR_MED_HEX, COLOUR_LOW_HEX,
)
from translation.formatter            import (
    char_results_to_html,
    build_annotated_text,
    result_to_markdown_table,
    summary_to_markdown,
    format_large_output,
)
from translation.text_builder         import TextBuilder, BuiltText

__all__ = [
    # dictionary
    "LETTERS", "NUMBERS", "PUNCTUATION", "MASTER_DICT",
    "CAPITAL_INDICATOR", "NUMBER_INDICATOR",
    "get_char", "is_known",
    # engine
    "TranslatorEngine", "TranslationResult", "CharResult",
    # confidence
    "tier_for", "analyse",
    "ConfidenceTier", "ConfidenceSummary",
    "COLOUR_HIGH_HEX", "COLOUR_MED_HEX", "COLOUR_LOW_HEX",
    # formatter
    "char_results_to_html", "build_annotated_text",
    "result_to_markdown_table", "summary_to_markdown",
    "format_large_output",
    # text builder
    "TextBuilder", "BuiltText",
]
