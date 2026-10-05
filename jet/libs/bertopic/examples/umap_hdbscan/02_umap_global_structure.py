"""
Advanced Demo: BERTopic with Global Structure Focus

Focuses on finding fewer, larger, general topics by using a high `n_neighbors` value in UMAP.

Uses jet.adapters.llama_cpp.embed_utils for embeddings and jet_telemetry for tracing.
Goal: Discover broad, general themes by focusing on global neighborhood structure.
"""

import numpy as np
from hdbscan import HDBSCAN
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet_telemetry import chain, embedding, get_trace_url, initialize_telemetry
from umap import UMAP

initialize_telemetry(service_name="bertopic-umap-global")

# Sample documents with distinct broad themes
docs = [
    "The stock market rose today due to tech earnings.",
    "Investors are worried about inflation rates.",
    "The new football season starts next week.",
    "The basketball team won the championship.",
    "Climate change is affecting polar ice caps.",
    "Renewable energy sources are becoming cheaper.",
    "Apple released a new iPhone with AI features.",
    "Google updated its search algorithm.",
    "The Olympics will be held in Paris.",
    "Soccer fans are excited for the World Cup.",
] * 5  # Expand to 50 docs


@embedding(model_name=EMBED_MODEL)
def get_embeddings(texts: list[str]) -> np.ndarray:
    """Jet-powered embedding with telemetry."""
    return embed(texts, model=EMBED_MODEL, return_format="numpy")


@chain(name="bertopic-global-structure-pipeline")
def run_global_topic_modeling(documents: list[str]):
    print("📊 Generating embeddings...")
    embeddings = get_embeddings(documents)

    print("⚙️ Configuring UMAP for GLOBAL structure (n_neighbors=30)...")
    # High n_neighbors focuses on broader global structure
    umap_model = UMAP(
        n_neighbors=30,  # High: Focus on global structure
        n_components=5,
        min_dist=0.0,
        metric="cosine",
    )

    # Standard HDBSCAN
    hdbscan_model = HDBSCAN(
        min_cluster_size=5, metric="euclidean", cluster_selection_method="eom"
    )

    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True
    )

    print("🚀 Fitting BERTopic...")
    topics, probs = topic_model.fit_transform(documents, embeddings)

    # Get topic info
    topic_info = topic_model.get_topic_info()
    print("\n🏆 Discovered Topics (Global Focus):")
    print(topic_info.head(10))

    return topic_model, topics


if __name__ == "__main__":
    model, topics = run_global_topic_modeling(docs)
    print(f"\n🔗 Trace: {get_trace_url()}")
