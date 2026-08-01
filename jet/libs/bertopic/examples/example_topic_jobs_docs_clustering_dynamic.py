"""
Demo: Run BERTopic pipeline with sample documents.

Demonstrates:
1. Default dynamic scaling with quality metrics
2. Custom UMAP with dynamic HDBSCAN and vectorizer
3. Override dynamic with fixed values + outlier threshold + topic floor
"""

from jet.libs.bertopic.examples.doc_samples import DOCS_LG
from jet.libs.bertopic.topic_docs_clustering_dynamic import run_bertopic_pipeline

print("=" * 70)
print("BERTopic Pipeline Demo with Dynamic Parameter Scaling")
print("=" * 70)

# Example 1: Fully dynamic parameters
print("\n" + "=" * 70)
print("Example 1: Fully Dynamic Parameters")
print("=" * 70)

result = run_bertopic_pipeline(
    documents=DOCS_LG,
    verbose=True,
)

print(f"\nFully dynamic pipeline found {len(result['topic_results'])} topics")
print("\nQuality Summary:")
for topic in result["topic_results"]:
    if topic["topic_id"] != -1:
        print(
            f"  {topic['name']}: "
            f"coherence={topic['coherence_score']:.4f}, "
            f"diversity={topic['keyword_diversity']:.2f}"
        )

# Example 2: Custom UMAP, Dynamic HDBSCAN & Vectorizer
print("\n" + "=" * 70)
print("Example 2: Custom UMAP, Dynamic HDBSCAN & Vectorizer")
print("=" * 70)

custom_result = run_bertopic_pipeline(
    documents=DOCS_LG,
    umap_params={
        "n_neighbors": 10,
        "n_components": 3,
    },
    verbose=True,
)

print(f"\nCustom UMAP pipeline found {len(custom_result['topic_results'])} topics")

# Example 3: Override Dynamic with Fixed Values + Post-Processing
print("\n" + "=" * 70)
print("Example 3: Fixed Values + Outlier Threshold + Topic Floor")
print("=" * 70)

fixed_result = run_bertopic_pipeline(
    documents=DOCS_LG,
    verbose=True,
)

print(f"\nFixed params pipeline found {len(fixed_result['topic_results'])} topics")
print("\nQuality Summary:")
for topic in fixed_result["topic_results"]:
    status = "OUTLIER" if topic["topic_id"] == -1 else f"Topic {topic['topic_id']}"
    if topic["coherence_score"] is not None:
        print(
            f"  {status} '{topic['name']}': "
            f"coherence={topic['coherence_score']:.4f}, "
            f"diversity={topic['keyword_diversity']:.2f}, "
            f"size={topic['size']}"
        )
