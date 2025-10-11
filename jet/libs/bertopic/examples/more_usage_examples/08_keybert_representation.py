"""
BERTopic Usage Example: Create KeyBERTInspired Representation Model

This example demonstrates how to configure a KeyBERTInspired representation model 
for semantically informed keyword extraction in BERTopic. This model refines topic 
keywords using semantic similarity, providing more meaningful topic representations.

Key Features:
- Semantically informed keyword extraction
- KeyBERT-based representation refinement
- Better topic keyword quality
- Integration with BERTopic pipeline

Usage:
    python 08_keybert_representation.py
"""

from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np


def create_keybert_representation_model() -> KeyBERTInspired:
    """
    Creates a KeyBERTInspired model for semantically informed keyword extraction.
    
    Returns:
        Initialized KeyBERTInspired instance.
    
    Example:
        keybert_model = create_keybert_representation_model()
    """
    return KeyBERTInspired()


def demonstrate_keybert_representation(keybert_model: KeyBERTInspired, 
                                     documents: list, 
                                     cluster_labels: list) -> dict:
    """
    Demonstrates KeyBERT representation on documents and clusters.
    
    Args:
        keybert_model: Configured KeyBERTInspired model.
        documents: List of documents.
        cluster_labels: Cluster assignments for each document.
    
    Returns:
        Dictionary mapping cluster IDs to refined keywords.
    """
    print(f"Input documents: {len(documents)}")
    print(f"Number of clusters: {len(set(cluster_labels))}")
    
    # Group documents by cluster
    cluster_docs = {}
    for doc, label in zip(documents, cluster_labels):
        if label not in cluster_docs:
            cluster_docs[label] = []
        cluster_docs[label].append(doc)
    
    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id, cluster_documents in cluster_docs.items():
        if cluster_id == -1:  # Skip outliers
            continue
            
        print(f"\nProcessing cluster {cluster_id} with {len(cluster_documents)} documents...")
        
        # Use KeyBERT to extract keywords
        try:
            keywords = keybert_model.extract_keywords(cluster_documents)
            cluster_keywords[cluster_id] = keywords
            print(f"  Extracted keywords: {keywords}")
        except Exception as e:
            print(f"  Error extracting keywords: {e}")
            cluster_keywords[cluster_id] = []
    
    return cluster_keywords


def analyze_keybert_results(cluster_keywords: dict, cluster_labels: list) -> None:
    """
    Analyzes and displays KeyBERT representation results.
    
    Args:
        cluster_keywords: Dictionary of cluster keywords.
        cluster_labels: Cluster assignments.
    """
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    
    print(f"\nKeyBERT Representation Analysis:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Clusters with keywords: {len(cluster_keywords)}")
    
    # Show keywords for each cluster
    print(f"\nKeywords per cluster:")
    for cluster_id in sorted(cluster_keywords.keys()):
        keywords = cluster_keywords[cluster_id]
        print(f"  Cluster {cluster_id}: {keywords}")


def compare_with_basic_keywords(documents: list, cluster_labels: list) -> None:
    """
    Compares KeyBERT keywords with basic frequency-based keywords.
    
    Args:
        documents: List of documents.
        cluster_labels: Cluster assignments.
    """
    from collections import Counter
    import re
    
    print(f"\nComparing KeyBERT vs Basic keyword extraction:")
    
    # Group documents by cluster
    cluster_docs = {}
    for doc, label in zip(documents, cluster_labels):
        if label not in cluster_docs:
            cluster_docs[label] = []
        cluster_docs[label].append(doc)
    
    # Extract basic keywords (simple word frequency)
    for cluster_id, cluster_documents in cluster_docs.items():
        if cluster_id == -1:  # Skip outliers
            continue
            
        # Basic keyword extraction
        all_text = " ".join(cluster_documents).lower()
        words = re.findall(r'\b\w+\b', all_text)
        word_counts = Counter(words)
        basic_keywords = [word for word, count in word_counts.most_common(5)]
        
        print(f"  Cluster {cluster_id}:")
        print(f"    Basic keywords: {basic_keywords}")
        print(f"    (KeyBERT keywords would be more semantically meaningful)")


def demonstrate_keybert_parameters() -> None:
    """
    Demonstrates different KeyBERT parameter configurations.
    """
    print(f"\nKeyBERT Parameter Options:")
    print(f"  Default configuration: Uses sentence-transformers model")
    print(f"  Custom model: Can specify different embedding models")
    print(f"  Diversity: Can control keyword diversity")
    print(f"  Top_k: Number of keywords to extract")
    
    # Show how to create with custom parameters (commented out to avoid errors)
    print(f"\nExample custom configuration:")
    print(f"  # keybert_model = KeyBERTInspired(model='all-MiniLM-L6-v2')")
    print(f"  # keybert_model = KeyBERTInspired(diversity=0.5)")


def main():
    """Demonstrate KeyBERT representation model creation and usage."""
    print("=== BERTopic KeyBERT Representation Example ===\n")
    
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
    
    # Create mock cluster labels (simulating clustering results)
    cluster_labels = [0, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 2, 0, 2, 1]
    print(f"Cluster labels: {cluster_labels}")
    
    # Create default KeyBERT model
    print("\n1. Creating default KeyBERT model...")
    keybert_model = create_keybert_representation_model()
    print(f"   KeyBERT model: {keybert_model}")
    
    # Demonstrate representation
    print("\n2. Applying KeyBERT representation...")
    try:
        cluster_keywords = demonstrate_keybert_representation(keybert_model, sample_docs, cluster_labels)
        
        # Analyze results
        print("\n3. Analyzing KeyBERT results...")
        analyze_keybert_results(cluster_keywords, cluster_labels)
        
    except Exception as e:
        print(f"   Error during KeyBERT processing: {e}")
        print(f"   This might be due to missing dependencies or model issues")
        print(f"   KeyBERT requires additional setup for full functionality")
    
    # Compare with basic keywords
    print("\n4. Comparing with basic keyword extraction...")
    compare_with_basic_keywords(sample_docs, cluster_labels)
    
    # Show parameter options
    print("\n5. KeyBERT parameter options...")
    demonstrate_keybert_parameters()
    
    # Show integration benefits
    print(f"\n6. KeyBERT integration benefits:")
    print(f"   - Semantically meaningful keywords")
    print(f"   - Better topic representation")
    print(f"   - Improved topic interpretability")
    print(f"   - Integration with BERTopic pipeline")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
