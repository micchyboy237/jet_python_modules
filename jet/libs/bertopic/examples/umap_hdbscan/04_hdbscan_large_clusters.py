"""
Advanced Demo: BERTopic with Large Clusters

Focuses on creating fewer, larger, general topics by using a high `min_cluster_size` in HDBSCAN.

Uses jet.adapters.llama_cpp.embed_utils for embeddings and jet_telemetry for tracing.
Goal: Discover broad, general themes by requiring more documents per topic.
"""

import numpy as np
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry
from umap import UMAP

initialize_telemetry(service_name="bertopic-hdbscan-large")

# Sample documents with broad themes
docs = [
    "The stock market rose today due to tech earnings.",
    "Investors are worried about inflation rates.",
    "Federal reserve interest rates impact borrowing.",
    "The new football season starts next week.",
    "The basketball team won the championship.",
    "Soccer fans are excited for the World Cup.",
    "Climate change is affecting polar ice caps.",
    "Renewable energy sources are becoming cheaper.",
    "Carbon emissions must be reduced globally.",
    "Apple released a new iPhone with AI features.",
    "Google updated its search algorithm.",
    "Microsoft launched a new cloud service.",
] * 5  # 60 docs


@embedding(model_name=EMBED_MODEL)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Jet-powered embedding with telemetry."""
    return embed(texts, model=EMBED_MODEL, return_format="numpy")


@chain(name="bertopic-large-clusters-pipeline")
def run_large_cluster_modeling(documents: list[str]):
    print("📊 Generating embeddings...")
    embeddings = get_embeddings(documents)

    # Standard UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")

    print("⚙️ Configuring HDBSCAN for LARGE clusters (min_cluster_size=15)...")
    # High min_cluster_size forces broader topics
    hdbscan_model = HDBSCAN(
        min_cluster_size=15,  # High: Requires more docs per topic
        metric="euclidean",
        cluster_selection_method="eom",
    )

    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True
    )

    print("🚀 Fitting BERTopic...")
    topics, probs = topic_model.fit_transform(documents, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("\n🏆 Discovered Topics (Large Clusters):")
    print(topic_info)

    return topic_model, topics


if __name__ == "__main__":
    model, topics = run_large_cluster_modeling(docs)
    print(f"\n🔗 Trace: {get_trace_url()}")
