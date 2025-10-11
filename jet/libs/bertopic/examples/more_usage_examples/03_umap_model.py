"""
BERTopic Usage Example: Create UMAP Dimensionality Reduction Model

This example demonstrates how to configure UMAP for reducing embedding dimensions 
while preserving the structure of the data. UMAP is crucial for BERTopic as it 
reduces high-dimensional embeddings to a lower-dimensional space suitable for clustering.

Key Features:
- Configurable UMAP parameters for different use cases
- Balance between local and global structure preservation
- Reproducible results with random state
- Integration with BERTopic pipeline

Usage:
    python 03_umap_model.py
"""

import umap
import numpy as np
from sentence_transformers import SentenceTransformer


def create_umap_model(n_neighbors: int = 5, min_dist: float = 0.01, random_state: int = 0) -> umap.UMAP:
    """
    Creates a UMAP model for dimensionality reduction.
    
    Args:
        n_neighbors: Balances local vs. global structure (lower = more local clusters).
        min_dist: Minimum distance between points (lower = tighter clusters).
        random_state: Seed for reproducibility.
    
    Returns:
        Initialized UMAP instance.
    
    Example:
        umap_model = create_umap_model(n_neighbors=5, min_dist=0.01)
    """
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )


def demonstrate_umap_reduction(umap_model: umap.UMAP, embeddings: np.ndarray) -> np.ndarray:
    """
    Demonstrates UMAP dimensionality reduction on embeddings.
    
    Args:
        umap_model: Configured UMAP model.
        embeddings: High-dimensional embeddings to reduce.
    
    Returns:
        Reduced embeddings in 2D space.
    """
    print(f"Original embeddings shape: {embeddings.shape}")
    
    # Fit and transform embeddings
    reduced_embeddings = umap_model.fit_transform(embeddings)
    
    print(f"Reduced embeddings shape: {reduced_embeddings.shape}")
    print(f"Dimensionality reduction: {embeddings.shape[1]} -> {reduced_embeddings.shape[1]}")
    
    return reduced_embeddings


def compare_umap_configurations(embeddings: np.ndarray) -> None:
    """
    Compares different UMAP configurations and their effects.
    
    Args:
        embeddings: High-dimensional embeddings to test.
    """
    configurations = [
        {"n_neighbors": 5, "min_dist": 0.01, "name": "Default (local focus)"},
        {"n_neighbors": 15, "min_dist": 0.1, "name": "Global focus"},
        {"n_neighbors": 3, "min_dist": 0.0, "name": "Very local"},
        {"n_neighbors": 30, "min_dist": 0.3, "name": "Very global"}
    ]
    
    print("\nComparing UMAP configurations:")
    for config in configurations:
        umap_model = create_umap_model(
            n_neighbors=config["n_neighbors"],
            min_dist=config["min_dist"]
        )
        reduced = umap_model.fit_transform(embeddings)
        
        # Calculate spread of points
        spread = np.std(reduced, axis=0)
        print(f"  {config['name']}: spread = {spread[0]:.3f}, {spread[1]:.3f}")


def main():
    """Demonstrate UMAP model creation and usage."""
    print("=== BERTopic UMAP Model Example ===\n")
    
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
        "Federal Reserve considers interest rate changes."
    ]
    
    # Create embeddings first
    print("1. Creating embeddings...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(sample_docs)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    # Create default UMAP model
    print("\n2. Creating default UMAP model...")
    umap_model = create_umap_model()
    print(f"   UMAP model: {umap_model}")
    print(f"   n_neighbors: {umap_model.n_neighbors}")
    print(f"   min_dist: {umap_model.min_dist}")
    
    # Demonstrate reduction
    print("\n3. Applying UMAP reduction...")
    reduced_embeddings = demonstrate_umap_reduction(umap_model, embeddings)
    
    # Show reduced embedding statistics
    print(f"\n4. Reduced embedding statistics:")
    print(f"   Mean X: {np.mean(reduced_embeddings[:, 0]):.4f}")
    print(f"   Mean Y: {np.mean(reduced_embeddings[:, 1]):.4f}")
    print(f"   Std X: {np.std(reduced_embeddings[:, 0]):.4f}")
    print(f"   Std Y: {np.std(reduced_embeddings[:, 1]):.4f}")
    
    # Compare different configurations
    print("\n5. Comparing UMAP configurations...")
    compare_umap_configurations(embeddings)
    
    # Show parameter effects
    print(f"\n6. UMAP parameter effects:")
    print(f"   n_neighbors: Lower values focus on local structure")
    print(f"   min_dist: Lower values create tighter clusters")
    print(f"   random_state: Ensures reproducible results")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
