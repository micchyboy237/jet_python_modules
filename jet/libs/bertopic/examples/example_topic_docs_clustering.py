"""
Demo: Run BERTopic pipeline with sample documents.

Demonstrates:
1. Default parameters with quality scoring
2. Outlier threshold to filter low-confidence assignments
3. Topic size floor to dissolve micro-clusters
"""

from jet.libs.bertopic.examples.doc_samples import DOCS_LG
from jet.libs.bertopic.topic_docs_clustering import run_bertopic_pipeline

print("=" * 70)
print("BERTopic Pipeline Demo with Quality Scoring & Post-Processing")
print("=" * 70)

# Example 1: Default parameters with quality scoring
print("\n" + "=" * 70)
print("Example 1: Default Parameters with Quality Metrics")
print("=" * 70)

result = run_bertopic_pipeline(
    documents=DOCS_LG,
    verbose=True,
)

print("\n" + "=" * 70)
print("Quality Summary:")
for topic in result["topic_results"]:
    if topic["topic_id"] != -1:
        print(
            f"  {topic['name']}: "
            f"coherence={topic['coherence_score']:.4f}, "
            f"diversity={topic['keyword_diversity']:.2f}, "
            f"size={topic['size']}"
        )

# Example 2: With outlier threshold
print("\n" + "=" * 70)
print("Example 2: Outlier Threshold (0.3) + Topic Floor (3)")
print("=" * 70)

filtered_result = run_bertopic_pipeline(
    documents=DOCS_LG,
    bertopic_params={
        "outlier_threshold": 0.3,
        "min_topic_floor": 3,
    },
    verbose=True,
)

print(
    f"\nFiltered pipeline: {len(filtered_result['topic_results'])} topics "
    f"({sum(1 for t in filtered_result['topic_results'] if t['topic_id'] == -1)} outliers)"
)
