# language.py

from typing import Literal, Optional, TypedDict

from fast_langdetect import detect, LangDetectConfig


class DetectLangResult(TypedDict):
    lang: str
    score: float


def detect_lang(
    text: str,
    *,
    model: Literal["lite", "full", "auto"] = "lite",
    max_input_length: Optional[int] = None,
    threshold: float = 0.1,
) -> DetectLangResult:
    """
    Detects the primary language of the input text using fast-langdetect.

    Returns the top language (ISO 639-1 code, lowercase) and its confidence score.
    Falls back to {"lang": "unknown", "score": 0.0} in these cases:
      - empty/meaningless input
      - detection exception
      - top score below threshold

    Args:
        text: Any string (will be lightly cleaned)
        model: Which FastText model to use
               - "lite"   → bundled, fastest, lowest memory (default)
               - "full"   → downloaded (~126 MB), highest accuracy
               - "auto"   → tries "full", falls back to "lite" on MemoryError
        max_input_length: Max characters to feed the model (None = no truncation)
                         200 is a good balance — longer helps accuracy, but >512 rarely needed
        threshold: If top score < this value → return "unknown"
                   (prevents misleading high-confidence on junk/short text)

    Returns:
        DetectLangResult dict with 'lang' and 'score'
    """
    if not text or not text.strip():
        return {"lang": "unknown", "score": 0.0}

    # ── Very light cleaning ───────────────────────────────────────────────
    # Keep punctuation & emojis — they help detection
    # Only normalize whitespace and strip
    cleaned = " ".join(text.split())

    # Configure detection, including model input truncation
    config = LangDetectConfig(max_input_length=max_input_length)

    try:
        # Always ask for top-1
        results = detect(
            cleaned,
            model=model,
            k=1,
            threshold=threshold,
            config=config,
        )

        if not results:
            return {"lang": "unknown", "score": 0.0}

        top = results[0]
        lang = top["lang"].lower()      # already lowercase in recent versions, but safe
        score = float(top["score"])

        if score < threshold:
            return {"lang": "unknown", "score": score}

        return {"lang": lang, "score": score}

    except Exception:  # noqa: BLE001
        # MemoryError, ValueError, etc. → graceful fallback
        # You can log here if you have a logger
        return {"lang": "unknown", "score": 0.0}