"""
Advanced Demo: BERTopic with Local Structure Focus

Focuses on finding many small, specific topics by using a low `n_neighbors` value in UMAP.

Uses jet.adapters.llama_cpp.embed_utils for embeddings and jet_telemetry for tracing.
Goal: Discover granular, specific topics by focusing on local neighborhood structure.
"""

import numpy as np
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry
from umap import UMAP

initialize_telemetry(service_name="bertopic-umap-local")

# Sample documents with subtle differences
docs = [
    "Python is great for data science and machine learning.",
    "Java is widely used in enterprise backend systems.",
    "JavaScript powers modern web frontends and React apps.",
    "Python scripts automate daily tasks and DevOps pipelines.",
    "C++ is essential for high-performance game engines.",
    "React components manage state in complex web applications.",
    "Spring Boot simplifies Java microservice development.",
    "DevOps tools like Docker and Kubernetes streamline deployment.",
    "Game physics engines require optimized C++ code.",
    "Data visualization libraries in Python help analyze trends.",
] * 5  # Expand to 50 docs for better clustering


@embedding(model_name=EMBED_MODEL)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Jet-powered embedding with telemetry."""
    return embed(texts, model=EMBED_MODEL, return_format="numpy")


@chain(name="bertopic-local-structure-pipeline")
def run_local_topic_modeling(documents: list[str]):
    print("📊 Generating embeddings...")
    embeddings = get_embeddings(documents)

    print("⚙️ Configuring UMAP for LOCAL structure (n_neighbors=5)...")
    # Low n_neighbors focuses on very local structure
    umap_model = UMAP(
        n_neighbors=5,  # Low: Focus on local details
        n_components=5,
        min_dist=0.0,
        metric="cosine",
    )

    # Standard HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=3,  # Allow small clusters
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
    print("\n🏆 Discovered Topics (Local Focus):")
    print(topic_info.head(10))

    return topic_model, topics


if __name__ == "__main__":
    model, topics = run_local_topic_modeling(docs)
    print(f"\n🔗 Trace: {get_trace_url()}")
