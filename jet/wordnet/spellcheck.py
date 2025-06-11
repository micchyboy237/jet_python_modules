from typing import List, Set, Optional, Tuple
from rapidfuzz import fuzz, process
from jet.logger import logger


def correct_typos(
    query_tokens: List[str],
    all_tokens: Set[str],
    threshold: float = 60.0,
    case_sensitive: bool = False,
    scorer: callable = fuzz.token_sort_ratio,
    max_corrections: Optional[int] = None,
    ignore_tokens: Optional[Set[str]] = None,
    return_details: bool = False
) -> List[str] | List[Tuple[str, str, float]]:
    """
    Correct typos in query tokens using fuzzy matching against a set of tokens.

    Args:
        query_tokens: List of query tokens to correct.
        all_tokens: Set of valid tokens to match against.
        threshold: Minimum similarity score (0-100) for a correction (default: 60.0).
        case_sensitive: Whether to perform case-sensitive matching (default: True).
        scorer: Fuzzy matching scorer from rapidfuzz (default: fuzz.token_sort_ratio).
        max_corrections: Maximum number of terms to correct (default: None, no limit).
        ignore_tokens: Set of tokens to skip correction for (default: None).
        return_details: If True, return list of (original, corrected, score) tuples (default: False).

    Returns:
        List of corrected tokens or list of (original, corrected, score) tuples if return_details is True.
    """
    logger.info(
        "Correcting typos for query tokens: %s (threshold=%.2f, case_sensitive=%s, scorer=%s, max_corrections=%s)",
        query_tokens, threshold, case_sensitive, scorer.__name__, max_corrections
    )

    if not query_tokens:
        logger.debug("Empty query tokens, returning empty list")
        return [] if not return_details else []

    if ignore_tokens is None:
        ignore_tokens = set()

    # Prepare tokens for case sensitivity
    working_tokens = all_tokens if case_sensitive else {
        t.lower() for t in all_tokens}
    logger.debug("Working tokens: %s", working_tokens)
    corrected_query = []
    corrections_count = 0

    for term in query_tokens:
        term_to_match = term if case_sensitive else term.lower()

        # Skip correction if term is in ignore_tokens or max corrections reached
        if term in ignore_tokens or (max_corrections is not None and corrections_count >= max_corrections):
            logger.debug(
                "Skipping correction for '%s' (ignore_tokens=%s, corrections_count=%d, max_corrections=%s)",
                term, term in ignore_tokens, corrections_count, max_corrections
            )
            corrected_query.append(
                (term, term, 100.0) if return_details else term)
            continue

        # Perform fuzzy matching
        matches = process.extract(
            term_to_match, working_tokens, scorer=scorer, limit=1)
        logger.debug("Typo correction for '%s': matches=%s", term, matches)

        if matches and matches[0][1] >= threshold:
            corrected_term = matches[0][0]
            score = matches[0][1]
            # Restore original case if case-sensitive is False
            if not case_sensitive:
                corrected_term = next(
                    t for t in all_tokens if t.lower() == corrected_term)
            logger.debug("Corrected '%s' to '%s' (score: %.2f, threshold: %.2f)",
                         term, corrected_term, score, threshold)
            corrections_count += 1
        else:
            corrected_term = term
            score = 100.0 if not matches else matches[0][1]
            logger.debug("No correction for '%s', keeping original (score: %.2f, threshold: %.2f)",
                         term, score, threshold)

        if return_details:
            corrected_query.append((term, corrected_term, score))
        else:
            corrected_query.append(corrected_term)

    logger.info("Corrected query: %s", corrected_query)
    return corrected_query
