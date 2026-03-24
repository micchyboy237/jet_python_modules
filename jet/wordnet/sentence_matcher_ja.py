from __future__ import annotations

import argparse
from typing import TypedDict

from rapidfuzz import fuzz, process


class FuzzyMatchResult(TypedDict):
    """TypedDict for the return value of fuzzy_shortest_best_match."""

    match: str
    score: float
    start: int
    end: int


def fuzzy_shortest_best_match(
    query: str,
    text: str,
    score_cutoff: int = 50,
    max_extra_chars: int = 20,
) -> FuzzyMatchResult:
    """
    Find the shortest contiguous substring in `text` that best matches `query`
    using fuzzy matching (WRatio), preferring higher score then shorter length.

    Args:
        query: The string to search for.
        text: The text to search within.
        score_cutoff: Minimum acceptable score (default: 50).
        max_extra_chars: Maximum extra characters allowed in the match window.

    Returns:
        FuzzyMatchResult containing match, score, start, and end indices.
    """
    if not query or not text:
        return {"match": "", "score": 0.0, "start": -1, "end": -1}

    # Step 1: Quick broad search
    candidates = process.extract(
        query, [text], scorer=fuzz.partial_ratio, limit=3, score_cutoff=score_cutoff
    )

    if not candidates:
        return {"match": "", "score": 0.0, "start": -1, "end": -1}

    # Step 2: Find shortest window with highest WRatio score
    best_score: float = -1.0
    best_start: int = -1
    best_end: int = -1
    best_match: str = ""
    best_length: int = float("inf")

    query_len = len(query)
    max_len = query_len + max_extra_chars

    for length in range(query_len, max_len + 1):
        for i in range(len(text) - length + 1):
            window = text[i : i + length]
            score = fuzz.WRatio(query, window)

            # Higher score wins; on tie, shorter length wins
            if score > best_score or (score == best_score and length < best_length):
                best_score = score
                best_start = i
                best_end = i + length
                best_match = window
                best_length = length

    # Fallback if no good match found
    if best_score < score_cutoff and candidates:
        best_match = candidates[0][0]
        best_score = float(candidates[0][1])
        best_start = text.find(best_match)
        best_end = best_start + len(best_match)

    return {
        "match": best_match,
        "score": best_score,
        "start": best_start,
        "end": best_end,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find the shortest best fuzzy match of a query inside a text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional required arguments
    parser.add_argument(
        "query",
        type=str,
        help="The query string to search for (e.g. '去る初めての消ひ狩りはえ')",
    )
    parser.add_argument(
        "text",
        type=str,
        help="The full text to search within",
    )

    # Optional keyword arguments with short flags
    parser.add_argument(
        "-c",
        "--score-cutoff",
        type=int,
        default=50,
        help="Minimum score to accept (0-100)",
    )
    parser.add_argument(
        "-e",
        "--max-extra-chars",
        type=int,
        default=20,
        help="Maximum extra characters allowed beyond query length",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output including highlighted text",
    )

    args = parser.parse_args()

    result: FuzzyMatchResult = fuzzy_shortest_best_match(
        query=args.query,
        text=args.text,
        score_cutoff=args.score_cutoff,
        max_extra_chars=args.max_extra_chars,
    )

    # Output results
    print(f"Match : {result['match']}")
    print(f"Score : {result['score']:.1f}")
    print(f"Slice : [{result['start']}:{result['end']}]")
    print(f"Length: {result['end'] - result['start']}")

    if result["score"] >= args.score_cutoff:
        print("✅ Accepted")
    else:
        print("❌ Below threshold")

    if args.verbose and result["start"] != -1:
        highlighted = (
            args.text[: result["start"]]
            + f"\033[1;33m{args.text[result['start'] : result['end']]}\033[0m"
            + args.text[result["end"] :]
        )
        print("\nHighlighted in text:")
        print(highlighted)


if __name__ == "__main__":
    main()
