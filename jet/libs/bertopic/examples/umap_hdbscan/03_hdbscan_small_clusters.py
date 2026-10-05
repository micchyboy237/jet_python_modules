"""
Advanced Demo: BERTopic with Small Clusters

Focuses on creating many small, specific topics by using a low `min_cluster_size` in HDBSCAN.

Uses jet.adapters.llama_cpp.embed_utils for embeddings and jet_telemetry for tracing.
Goal: Discover highly specific, niche topics even if they have few documents.
"""

import numpy as np
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry
from umap import UMAP

initialize_telemetry(service_name="bertopic-hdbscan-small")

# Sample documents with niche topics
docs = [
    "Quantum computing uses qubits for processing.",
    "Entanglement is a key feature of quantum mechanics.",
    "Shor's algorithm can break RSA encryption.",
    "My cat likes to sleep on the keyboard.",
    "Dog training requires positive reinforcement.",
    "Parrots can mimic human speech patterns.",
    "The new electric car has a 500-mile range.",
    "Battery technology is improving rapidly.",
    "Charging stations are becoming more common.",
    "SpaceX launched another Starlink satellite.",
] * 3  # Small dataset to demonstrate fragmentation risk


@embedding(model_name=EMBED_MODEL)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Jet-powered embedding with telemetry."""
    return embed(texts, model=EMBED_MODEL, return_format="numpy")


@chain(name="bertopic-small-clusters-pipeline")
def run_small_cluster_modeling(documents: list[str]):
    print("📊 Generating embeddings...")
    embeddings = get_embeddings(documents)

    # Standard UMAP
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")

    print("⚙️ Configuring HDBSCAN for SMALL clusters (min_cluster_size=3)...")
    # Low min_cluster_size allows small, specific topics
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,  # Low: Allows small clusters
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
    print("\n🏆 Discovered Topics (Small Clusters):")
    print(topic_info)

    return topic_model, topics


if __name__ == "__main__":
    model, topics = run_small_cluster_modeling(docs)
    print(f"\n🔗 Trace: {get_trace_url()}")
