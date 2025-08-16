from typing import TypedDict
from ftlangdetect import detect
from jet.utils.string_utils import remove_non_alpha_numeric


class DetectLangResult(TypedDict):
    lang: str  # Ex. "en", "tl"
    score: float


def detect_lang(text) -> DetectLangResult:
    text = remove_non_alpha_numeric(text)

    result = detect(text=text, low_memory=True)
    return {
        "lang": str(result["lang"]),
        "score": float(result["score"]),
    }
