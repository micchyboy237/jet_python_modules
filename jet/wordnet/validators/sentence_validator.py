def is_valid_sentence(text: str) -> bool:
    """
    Quick heuristic check whether a string should be considered a valid sentence.
    Returns True for things that look like proper sentences or meaningful fragments.
    Returns False for junk, headers, very short garbage, etc.
    """
    if not text:
        return False

    cleaned = text.strip()
    if not cleaned:
        return False

    # Remove NEWLINE_TOKEN for validation purposes
    cleaned_for_check = cleaned.replace("NEWLINE_TOKEN", " ")

    if len(cleaned_for_check) <= 2:
        return False

    # Allow short punctuated fragments
    if len(cleaned_for_check) <= 6:
        if cleaned_for_check.rstrip().endswith((".", "!", "?")):
            return True
        return False

    words = cleaned_for_check.split()

    # Reject single very long "words"
    if len(words) == 1 and len(cleaned_for_check) > 40:
        return False

    # Reject ALL CAPS short strings (likely headers)
    if cleaned_for_check.isupper() and len(cleaned_for_check) < 60:
        return False

    # Reject common document structure labels
    if cleaned_for_check.lower() in {
        "abstract",
        "introduction",
        "conclusion",
        "references",
        "table of contents",
        "acknowledgements",
        "appendix",
        "chapter",
        "figure",
        "table",
        "section",
        "page",
        "isbn",
        "doi",
    }:
        return False

    # Handle list markers more carefully
    if cleaned_for_check.lstrip().startswith(("•", "○", "→", "▪", "*", "-", "–", "—")):
        if len(words) <= 5:
            return False

    import re

    # Reject lines with excessive special characters
    if re.search(r"[-_=]{4,}", cleaned_for_check):
        return False
    if re.search(r"\W{6,}", cleaned_for_check):
        return False

    # Must end with proper sentence terminator OR be a meaningful phrase
    if cleaned_for_check.rstrip().endswith((".", "!", "?", '."', '."', ".”", "…")):
        return True

    # For non-terminated text, require minimum structure
    has_space = " " in cleaned_for_check
    has_letter = any(c.isalpha() for c in cleaned_for_check)

    # Require both letters AND spaces AND minimum word count for non-terminated text
    if has_letter and has_space and len(words) >= 2:
        return True

    return False
