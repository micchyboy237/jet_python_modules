"""
BERTopic Usage Example: Create K-Means Clustering Model

This example demonstrates how to configure K-Means as an alternative to HDBSCAN 
for clustering in BERTopic. K-Means is useful when you want a fixed number of topics 
and prefer spherical, well-separated clusters.

Key Features:
- Fixed number of clusters (topics)
- Spherical cluster shapes
- Deterministic results
- Alternative to HDBSCAN for specific use cases

Usage:
    python 05_kmeans_model.py
"""

import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import umap


def create_kmeans_model(n_clusters: int = 8, random_state: int = 0) -> KMeans:
    """
    Creates a K-Means model for partitioning into a fixed number of topics.
    
    Args:
        n_clusters: Exact number of topics/clusters.
        random_state: Seed for reproducibility.
    
    Returns:
        Initialized KMeans instance.
    
    Example:
        kmeans_model = create_kmeans_model(n_clusters=8)
    """
    return KMeans(n_clusters=n_clusters, random_state=random_state)


def demonstrate_kmeans_clustering(kmeans_model: KMeans, embeddings: np.ndarray) -> np.ndarray:
    """
    Demonstrates K-Means clustering on embeddings.
    
    Args:
        kmeans_model: Configured K-Means model.
        embeddings: Reduced embeddings to cluster.
    
    Returns:
        Cluster labels for each embedding.
    """
    print(f"Input embeddings shape: {embeddings.shape}")
    print(f"Number of clusters: {kmeans_model.n_clusters}")
    
    # Fit and predict clusters
    cluster_labels = kmeans_model.fit_predict(embeddings)
    
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Number of clusters found: {len(set(cluster_labels))}")
    
    return cluster_labels


def analyze_kmeans_results(cluster_labels: np.ndarray, documents: list) -> None:
    """
    Analyzes and displays K-Means clustering results.
    
    Args:
        cluster_labels: Cluster assignments for each document.
        documents: Original documents.
    """
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels)
    
    print(f"\nK-Means Clustering Analysis:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Number of clusters: {n_clusters}")
    
    # Show documents per cluster
    print(f"\nDocuments per cluster:")
    for label in sorted(unique_labels):
        count = np.sum(cluster_labels == label)
        print(f"  Cluster {label}: {count} documents")
        
        # Show sample documents from this cluster
        cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == label]
        for i, doc in enumerate(cluster_docs[:2]):  # Show first 2 docs
            print(f"    - {doc}")


def compare_kmeans_configurations(embeddings: np.ndarray) -> None:
    """
    Compares different K-Means configurations and their effects.
    
    Args:
        embeddings: Reduced embeddings to test.
    """
    configurations = [
        {"n_clusters": 3, "name": "Few clusters"},
        {"n_clusters": 5, "name": "Medium clusters"},
        {"n_clusters": 8, "name": "Many clusters"},
        {"n_clusters": 12, "name": "Very many clusters"}
    ]
    
    print("\nComparing K-Means configurations:")
    for config in configurations:
        kmeans_model = create_kmeans_model(n_clusters=config["n_clusters"])
        labels = kmeans_model.fit_predict(embeddings)
        
        n_clusters = len(set(labels))
        print(f"  {config['name']}: {n_clusters} clusters")
        
        # Show cluster sizes
        cluster_sizes = [np.sum(labels == i) for i in range(n_clusters)]
        print(f"    Cluster sizes: {cluster_sizes}")


def compare_kmeans_vs_hdbscan(embeddings: np.ndarray) -> None:
    """
    Compares K-Means vs HDBSCAN clustering approaches.
    
    Args:
        embeddings: Reduced embeddings to test.
    """
    import hdbscan
    
    print("\nComparing K-Means vs HDBSCAN:")
    
    # K-Means approach
    kmeans_model = create_kmeans_model(n_clusters=5)
    kmeans_labels = kmeans_model.fit_predict(embeddings)
    kmeans_clusters = len(set(kmeans_labels))
    
    # HDBSCAN approach
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    hdbscan_labels = hdbscan_model.fit_predict(embeddings)
    hdbscan_clusters = len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)
    hdbscan_outliers = np.sum(hdbscan_labels == -1)
    
    print(f"  K-Means: {kmeans_clusters} clusters, 0 outliers")
    print(f"  HDBSCAN: {hdbscan_clusters} clusters, {hdbscan_outliers} outliers")
    print(f"  K-Means: Fixed number of clusters, no outliers")
    print(f"  HDBSCAN: Variable clusters, can identify outliers")


def main():
    """Demonstrate K-Means model creation and usage."""
    print("=== BERTopic K-Means Model Example ===\n")
    
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
    
    # Create embeddings and reduce dimensions
    print("1. Creating embeddings and reducing dimensions...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(sample_docs)
    
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    reduced_embeddings = umap_model.fit_transform(embeddings)
    print(f"   Reduced embeddings shape: {reduced_embeddings.shape}")
    
    # Create default K-Means model
    print("\n2. Creating default K-Means model...")
    kmeans_model = create_kmeans_model()
    print(f"   K-Means model: {kmeans_model}")
    print(f"   n_clusters: {kmeans_model.n_clusters}")
    
    # Demonstrate clustering
    print("\n3. Applying K-Means clustering...")
    cluster_labels = demonstrate_kmeans_clustering(kmeans_model, reduced_embeddings)
    
    # Analyze results
    print("\n4. Analyzing clustering results...")
    analyze_kmeans_results(cluster_labels, sample_docs)
    
    # Compare different configurations
    print("\n5. Comparing K-Means configurations...")
    compare_kmeans_configurations(reduced_embeddings)
    
    # Compare with HDBSCAN
    print("\n6. Comparing K-Means vs HDBSCAN...")
    compare_kmeans_vs_hdbscan(reduced_embeddings)
    
    # Show parameter effects
    print(f"\n7. K-Means parameter effects:")
    print(f"   n_clusters: Exact number of topics to create")
    print(f"   random_state: Ensures reproducible results")
    print(f"   Use when: You want a fixed number of topics")
    print(f"   Avoid when: You need outlier detection or variable cluster sizes")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
