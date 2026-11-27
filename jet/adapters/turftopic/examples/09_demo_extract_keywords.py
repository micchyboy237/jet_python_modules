"""
Demo 9: Keyword Extraction

You can do this step manually if you want to precompute the keyword matrix.
Keywords are represented as dictionaries mapping words to keyword importances.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from mock import SAMPLE_CORPUS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF


def format_keywords(keywords, top_n=10):
    """
    Format keywords into a human-readable table.

    Args:
        keywords: List of dictionaries mapping words to importance scores
        top_n: Number of top keywords to display per topic

    Returns:
        Formatted string representation of keywords
    """
    if not keywords:
        return "No keywords extracted."

    lines = []
    lines.append(f"{'=' * 80}")
    lines.append(f"EXTRACTED TOPICS ({len(keywords)} topics)")
    lines.append(f"{'=' * 80}")

    for i, topic_keywords in enumerate(keywords, 1):
        # Sort keywords by importance (descending)
        sorted_keywords = sorted(
            topic_keywords.items(), key=lambda x: float(x[1]), reverse=True
        )

        # Take top N keywords
        top_keywords = sorted_keywords[:top_n]

        lines.append(f"\n📌 Topic {i}:")
        lines.append(f"{'-' * 80}")

        # Create a formatted table
        header = f"  {'Rank':<6} {'Keyword':<25} {'Importance':>12}"
        lines.append(header)
        lines.append(f"  {'─' * 6} {'─' * 25} {'─' * 12}")

        for rank, (keyword, score) in enumerate(top_keywords, 1):
            score_float = float(score)
            # Format score as percentage
            score_pct = f"{score_float:.2%}"
            lines.append(f"  {rank:<6} {keyword:<25} {score_pct:>12}")

    lines.append(f"\n{'=' * 80}")
    return "\n".join(lines)


def format_keywords_compact(keywords, top_n=5):
    """
    Format keywords in a compact, comma-separated format.

    Args:
        keywords: List of dictionaries mapping words to importance scores
        top_n: Number of top keywords to display per topic

    Returns:
        Compact string representation of keywords
    """
    if not keywords:
        return "No keywords extracted."

    lines = []
    lines.append("\n📊 Extracted Topics Summary:")
    lines.append("━" * 80)

    for i, topic_keywords in enumerate(keywords, 1):
        # Sort keywords by importance (descending)
        sorted_keywords = sorted(
            topic_keywords.items(), key=lambda x: float(x[1]), reverse=True
        )

        # Take top N keywords
        top_keywords = sorted_keywords[:top_n]

        # Format as: keyword1 (score), keyword2 (score), ...
        keyword_strs = [f"{kw} ({float(score):.3f})" for kw, score in top_keywords]

        lines.append(f"Topic {i:2d}: {', '.join(keyword_strs)}")

    lines.append("━" * 80)
    return "\n".join(lines)


print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model...")
# Initialize KeyNMF with 5 topics for this small dataset
# Use min_df=1 because the dataset is small (default is 10)
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(
    n_components=5,
    vectorizer=vectorizer,
    encoder=encoder,
)

print("\n--- Extract Topics ---")
keywords = model.extract_keywords(SAMPLE_CORPUS)

# Display in detailed table format
print(format_keywords(keywords, top_n=8))

# Optionally, also display compact format
# print(format_keywords_compact(keywords, top_n=5))

# Keep original raw output available if needed
# print("\nRaw keywords data:")
# print(keywords)
