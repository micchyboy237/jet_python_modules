"""
BERTopic Usage Example: Create Basic BERTopic Model

This example demonstrates how to create and fit a basic BERTopic model with 
optional custom components. This is the foundation for topic modeling, using 
default components or allowing customization of individual parts.

Key Features:
- Basic BERTopic initialization and fitting
- Optional custom components (embedding, UMAP, HDBSCAN)
- Topic assignment and probability extraction
- Foundation for advanced topic modeling

Usage:
    python 09_basic_topic_model.py
"""

from typing import List, Tuple, Optional
import umap
import hdbscan
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np


def create_basic_topic_model(
    embedding_model: Optional[SentenceTransformer] = None,
    umap_model: Optional[umap.UMAP] = None,
    hdbscan_model: Optional[hdbscan.HDBSCAN] = None,
    documents: Optional[List[str]] = None
) -> Tuple[BERTopic, Optional[List[int]], Optional[np.ndarray]]:
    """
    Creates and fits a basic BERTopic model (uses defaults if components not provided).
    
    Args:
        embedding_model: Optional custom embedding model.
        umap_model: Optional custom UMAP model.
        hdbscan_model: Optional custom HDBSCAN model.
        documents: Optional list of documents to fit on (if None, model is created but not fitted).
    
    Returns:
        Tuple of (fitted BERTopic model, topic assignments, topic probabilities).
    
    Example:
        emb_model = create_embedding_model()
        topics, probs = create_basic_topic_model(embedding_model=emb_model, documents=SAMPLE_DOCS)
        topic_info = topics.get_topic_info()
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )
    
    if documents is not None:
        topics, probs = topic_model.fit_transform(documents)
    else:
        topics, probs = None, None
    
    return topic_model, topics, probs


def demonstrate_basic_usage(documents: List[str]) -> None:
    """
    Demonstrates basic BERTopic usage with default components.
    
    Args:
        documents: List of documents to model.
    """
    print("1. Basic BERTopic with default components...")
    
    # Create model with defaults
    topic_model, topics, probs = create_basic_topic_model(documents=documents)
    
    print(f"   Model created and fitted")
    print(f"   Number of topics: {len(set(topics)) - (1 if -1 in topics else 0)}")
    print(f"   Number of outliers: {np.sum(np.array(topics) == -1)}")
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    print(f"   Topic info shape: {topic_info.shape}")
    
    return topic_model, topics, probs


def demonstrate_custom_components(documents: List[str]) -> None:
    """
    Demonstrates BERTopic with custom components.
    
    Args:
        documents: List of documents to model.
    """
    print("\n2. BERTopic with custom components...")
    
    # Create custom components
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    
    print(f"   Custom embedding model: {embedding_model}")
    print(f"   Custom UMAP model: {umap_model}")
    print(f"   Custom HDBSCAN model: {hdbscan_model}")
    
    # Create model with custom components
    topic_model, topics, probs = create_basic_topic_model(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        documents=documents
    )
    
    print(f"   Model created with custom components")
    print(f"   Number of topics: {len(set(topics)) - (1 if -1 in topics else 0)}")
    
    return topic_model, topics, probs


def demonstrate_kmeans_alternative(documents: List[str]) -> None:
    """
    Demonstrates BERTopic with K-Means instead of HDBSCAN.
    
    Args:
        documents: List of documents to model.
    """
    print("\n3. BERTopic with K-Means clustering...")
    
    # Create custom components with K-Means
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    kmeans_model = KMeans(n_clusters=5, random_state=0)
    
    print(f"   Using K-Means with {kmeans_model.n_clusters} clusters")
    
    # Create model with K-Means
    topic_model, topics, probs = create_basic_topic_model(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=kmeans_model,  # Note: BERTopic expects hdbscan_model parameter
        documents=documents
    )
    
    print(f"   Model created with K-Means")
    print(f"   Number of topics: {len(set(topics))}")
    
    return topic_model, topics, probs


def analyze_topic_results(topic_model: BERTopic, topics: List[int], documents: List[str]) -> None:
    """
    Analyzes and displays topic modeling results.
    
    Args:
        topic_model: Fitted BERTopic model.
        topics: Topic assignments.
        documents: Original documents.
    """
    print(f"\n4. Analyzing topic results...")
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    print(f"   Topic information:")
    print(topic_info)
    
    # Show topics and their documents
    unique_topics = sorted(set(topics))
    print(f"\n   Documents per topic:")
    for topic_id in unique_topics:
        if topic_id == -1:
            print(f"   Outliers: {np.sum(np.array(topics) == topic_id)} documents")
        else:
            topic_docs = [doc for i, doc in enumerate(documents) if topics[i] == topic_id]
            print(f"   Topic {topic_id}: {len(topic_docs)} documents")
            for i, doc in enumerate(topic_docs[:2]):  # Show first 2 docs
                print(f"     - {doc}")
    
    # Show topic keywords
    print(f"\n   Topic keywords:")
    for topic_id in unique_topics:
        if topic_id == -1:
            continue
        try:
            keywords = topic_model.get_topic(topic_id)
            if keywords:
                top_words = [word for word, score in keywords[:5]]
                print(f"   Topic {topic_id}: {top_words}")
        except Exception as e:
            print(f"   Topic {topic_id}: Error getting keywords - {e}")


def compare_model_configurations(documents: List[str]) -> None:
    """
    Compares different BERTopic model configurations.
    
    Args:
        documents: List of documents to test.
    """
    print(f"\n5. Comparing BERTopic configurations...")
    
    configurations = [
        {"name": "Default", "components": {}},
        {"name": "Custom UMAP", "components": {"umap_model": umap.UMAP(n_neighbors=3, min_dist=0.0)}},
        {"name": "Custom HDBSCAN", "components": {"hdbscan_model": hdbscan.HDBSCAN(min_cluster_size=5)}},
        {"name": "K-Means", "components": {"hdbscan_model": KMeans(n_clusters=4)}}
    ]
    
    for config in configurations:
        try:
            topic_model, topics, probs = create_basic_topic_model(
                documents=documents,
                **config["components"]
            )
            n_topics = len(set(topics)) - (1 if -1 in topics else 0)
            n_outliers = np.sum(np.array(topics) == -1)
            print(f"   {config['name']}: {n_topics} topics, {n_outliers} outliers")
        except Exception as e:
            print(f"   {config['name']}: Error - {e}")


def main():
    """Demonstrate basic BERTopic model creation and usage."""
    print("=== BERTopic Basic Model Example ===\n")
    
    # Sample documents for testing
    sample_docs = [
        "Apple stock rises amid market volatility.",
        "Berkshire Hathaway invests in tech startups.",
        "Nasdaq index hits new high on earnings reports.",
        "Tesla announces new battery technology.",
        "Inflation concerns impact bond yields.",
        "Crypto market rebounds after regulatory news.",
        "Amazon expands cloud services in Europe.",
        "Gold prices surge due to geopolitical tensions.",
        "Microsoft reports strong quarterly earnings.",
        "Federal Reserve considers interest rate changes.",
        "Bitcoin price reaches new all-time high.",
        "Oil prices fluctuate due to supply concerns.",
        "Tech sector shows mixed performance today.",
        "Banking stocks decline on regulatory news.",
        "Healthcare companies report positive results."
    ]
    
    print(f"Sample documents: {len(sample_docs)}")
    
    # Demonstrate basic usage
    topic_model, topics, probs = demonstrate_basic_usage(sample_docs)
    
    # Demonstrate custom components
    topic_model_custom, topics_custom, probs_custom = demonstrate_custom_components(sample_docs)
    
    # Demonstrate K-Means alternative
    topic_model_kmeans, topics_kmeans, probs_kmeans = demonstrate_kmeans_alternative(sample_docs)
    
    # Analyze results
    analyze_topic_results(topic_model, topics, sample_docs)
    
    # Compare configurations
    compare_model_configurations(sample_docs)
    
    # Show next steps
    print(f"\n6. Next steps for advanced usage:")
    print(f"   - Add custom vectorizer for better text processing")
    print(f"   - Use c-TF-IDF for improved topic keywords")
    print(f"   - Integrate KeyBERT for semantic keyword extraction")
    print(f"   - Visualize topics with topic_model.visualize_topics()")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
