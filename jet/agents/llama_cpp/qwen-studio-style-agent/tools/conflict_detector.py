import re
from typing import Any


def extract_numeric_facts(snippet_text: str) -> list[dict[str, Any]]:
    """Extract numeric claims with context from search snippets."""
    facts = []
    # Match patterns like "165 minutes", "2h 46m", "166 min", "$1.2 billion"
    patterns = [
        r"(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|seconds?|secs?)",
        r"(\d+)\s*h\s*(\d+)\s*m",
        r"\$(\d+(?:\.\d+)?)\s*(billion|million|trillion)",
        r"(\d+(?:\.\d+)?)%",
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, snippet_text, re.IGNORECASE):
            facts.append(
                {
                    "value": match.group(0),
                    "normalized": _normalize_value(match, pattern),
                    "context": snippet_text[
                        max(0, match.start() - 50) : match.end() + 50
                    ].strip(),
                }
            )
    return facts


def _normalize_value(match: re.Match, pattern: str) -> float | None:
    """Convert extracted value to comparable float."""
    try:
        if "minutes" in pattern or "mins" in pattern:
            return float(match.group(1))
        if r"h\s*(\d+)\s*m" in pattern:
            hours = float(match.group(1))
            mins = float(match.group(2)) if match.lastindex >= 2 else 0
            return hours * 60 + mins
        if "billion" in pattern.lower():
            return float(match.group(1)) * 1e9
        if "million" in pattern.lower():
            return float(match.group(1)) * 1e6
        if "%" in pattern:
            return float(match.group(1))
        return float(match.group(1))
    except (ValueError, IndexError):
        return None


def detect_conflicts(search_results: str, tolerance: float = 0.05) -> dict[str, Any]:
    """
    Analyze search result text for conflicting numeric claims.
    Returns conflict report with recommendation.
    """
    all_facts = extract_numeric_facts(search_results)

    if len(all_facts) < 2:
        return {
            "has_conflict": False,
            "reason": "insufficient_data",
            "facts_found": len(all_facts),
            "recommendation": "proceed_with_snippet",
        }

    # Group by normalized value clusters
    clusters: dict[float, list[dict]] = {}
    for fact in all_facts:
        if fact["normalized"] is None:
            continue
        placed = False
        for cluster_val in clusters:
            if abs(fact["normalized"] - cluster_val) / max(cluster_val, 1) <= tolerance:
                clusters[cluster_val].append(fact)
                placed = True
                break
        if not placed:
            clusters[fact["normalized"]] = [fact]

    # Conflict exists if multiple distinct clusters with significant support
    significant_clusters = {k: v for k, v in clusters.items() if len(v) >= 1}

    if len(significant_clusters) <= 1:
        return {
            "has_conflict": False,
            "reason": "consistent_values",
            "clusters": len(significant_clusters),
            "consensus_value": list(significant_clusters.keys())[0]
            if significant_clusters
            else None,
            "recommendation": "proceed_with_snippet",
        }

    # Multiple conflicting values detected
    return {
        "has_conflict": True,
        "reason": "multiple_distinct_values",
        "clusters": len(significant_clusters),
        "values": [
            {"value": k, "mentions": len(v), "sample": v[0]["value"]}
            for k, v in sorted(significant_clusters.items(), key=lambda x: -len(x[1]))
        ],
        "recommendation": "mandatory_extraction",
    }
