"""
BERTopic Usage Example: Create HDBSCAN Clustering Model

This example demonstrates how to configure HDBSCAN for density-based clustering 
of reduced embeddings. HDBSCAN is the default clustering algorithm in BERTopic 
as it can find clusters of varying densities and identify outliers.

Key Features:
- Density-based clustering for varying cluster shapes
- Automatic outlier detection
- Configurable cluster size and density parameters
- Integration with BERTopic pipeline

Usage:
    python 04_hdbscan_model.py
"""

import hdbscan
import numpy as np
import umap
from sentence_transformers import SentenceTransformer


def create_hdbscan_model(min_cluster_size: int = 15, min_samples: int = 5, random_state: int = 0) -> hdbscan.HDBSCAN:
    """
    Creates an HDBSCAN model for clustering reduced embeddings.
    
    Args:
        min_cluster_size: Minimum points per cluster (higher = fewer, denser topics).
        min_samples: Controls outlier detection (higher = more conservative).
        random_state: Seed for reproducibility.
    
    Returns:
        Initialized HDBSCAN instance.
    
    Example:
        hdbscan_model = create_hdbscan_model(min_cluster_size=15, min_samples=5)
    """
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        min_samples=min_samples
    )


def demonstrate_hdbscan_clustering(hdbscan_model: hdbscan.HDBSCAN, embeddings: np.ndarray) -> np.ndarray:
    """
    Demonstrates HDBSCAN clustering on embeddings.
    
    Args:
        hdbscan_model: Configured HDBSCAN model.
        embeddings: Reduced embeddings to cluster.
    
    Returns:
        Cluster labels for each embedding.
    """
    print(f"Input embeddings shape: {embeddings.shape}")
    
    # Fit and predict clusters
    cluster_labels = hdbscan_model.fit_predict(embeddings)
    
    print(f"Cluster labels shape: {cluster_labels.shape}")
    print(f"Number of clusters found: {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}")
    print(f"Number of outliers: {np.sum(cluster_labels == -1)}")
    
    return cluster_labels


def analyze_clustering_results(cluster_labels: np.ndarray, documents: list) -> None:
    """
    Analyzes and displays clustering results.
    
    Args:
        cluster_labels: Cluster assignments for each document.
        documents: Original documents.
    """
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_outliers = np.sum(cluster_labels == -1)
    
    print("\nClustering Analysis:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of outliers: {n_outliers}")
    print(f"  Outlier percentage: {n_outliers/len(documents)*100:.1f}%")
    
    # Show documents per cluster
    print("\nDocuments per cluster:")
    for label in sorted(unique_labels):
        if label == -1:
            print(f"  Outliers: {np.sum(cluster_labels == label)} documents")
        else:
            count = np.sum(cluster_labels == label)
            print(f"  Cluster {label}: {count} documents")
            
            # Show sample documents from this cluster
            cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == label]
            for i, doc in enumerate(cluster_docs[:2]):  # Show first 2 docs
                print(f"    - {doc}")


def compare_hdbscan_configurations(embeddings: np.ndarray) -> None:
    """
    Compares different HDBSCAN configurations and their effects.
    
    Args:
        embeddings: Reduced embeddings to test.
    """
    configurations = [
        {"min_cluster_size": 5, "min_samples": 3, "name": "Small clusters"},
        {"min_cluster_size": 15, "min_samples": 5, "name": "Default"},
        {"min_cluster_size": 30, "min_samples": 10, "name": "Large clusters"},
        {"min_cluster_size": 10, "min_samples": 15, "name": "Conservative"}
    ]
    
    print("\nComparing HDBSCAN configurations:")
    for config in configurations:
        hdbscan_model = create_hdbscan_model(
            min_cluster_size=config["min_cluster_size"],
            min_samples=config["min_samples"]
        )
        labels = hdbscan_model.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = np.sum(labels == -1)
        
        print(f"  {config['name']}: {n_clusters} clusters, {n_outliers} outliers")


def main():
    """Demonstrate HDBSCAN model creation and usage."""
    print("=== BERTopic HDBSCAN Model Example ===\n")
    
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
    
    # Create default HDBSCAN model
    print("\n2. Creating default HDBSCAN model...")
    hdbscan_model = create_hdbscan_model()
    print(f"   HDBSCAN model: {hdbscan_model}")
    print(f"   min_cluster_size: {hdbscan_model.min_cluster_size}")
    print(f"   min_samples: {hdbscan_model.min_samples}")
    
    # Demonstrate clustering
    print("\n3. Applying HDBSCAN clustering...")
    cluster_labels = demonstrate_hdbscan_clustering(hdbscan_model, reduced_embeddings)
    
    # Analyze results
    print("\n4. Analyzing clustering results...")
    analyze_clustering_results(cluster_labels, sample_docs)
    
    # Compare different configurations
    print("\n5. Comparing HDBSCAN configurations...")
    compare_hdbscan_configurations(reduced_embeddings)
    
    # Show parameter effects
    print("\n6. HDBSCAN parameter effects:")
    print("   min_cluster_size: Higher values create fewer, larger clusters")
    print("   min_samples: Higher values are more conservative about outliers")
    print("   metric: 'euclidean' works well with UMAP-reduced embeddings")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
