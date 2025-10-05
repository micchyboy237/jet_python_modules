"""
BERTopic Usage Example: Create c-TF-IDF Transformer Model

This example demonstrates how to configure a c-TF-IDF transformer for topic-level 
word weighting in BERTopic. The c-TF-IDF transformer applies TF-IDF weighting 
at the cluster level, helping to identify the most important words for each topic.

Key Features:
- Cluster-level TF-IDF weighting
- BM25 variant for handling document length variations
- Topic keyword importance scoring
- Integration with BERTopic pipeline

Usage:
    python 07_ctfidf_model.py
"""

from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def create_ctfidf_model(bm25_weighting: bool = True) -> ClassTfidfTransformer:
    """
    Creates a c-TF-IDF transformer for topic-level word weighting.
    
    Args:
        bm25_weighting: Use BM25 variant for handling document length variations.
    
    Returns:
        Initialized ClassTfidfTransformer instance.
    
    Example:
        ctfidf = create_ctfidf_model(bm25_weighting=True)
    """
    return ClassTfidfTransformer(bm25_weighting=bm25_weighting)


def demonstrate_ctfidf_transformer(ctfidf_model: ClassTfidfTransformer, 
                                doc_term_matrix, 
                                cluster_labels: list) -> np.ndarray:
    """
    Demonstrates c-TF-IDF transformation on document-term matrix.
    
    Args:
        ctfidf_model: Configured c-TF-IDF transformer.
        doc_term_matrix: Document-term matrix from CountVectorizer.
        cluster_labels: Cluster assignments for each document.
    
    Returns:
        c-TF-IDF weighted matrix.
    """
    print(f"Input document-term matrix shape: {doc_term_matrix.shape}")
    print(f"Number of clusters: {len(set(cluster_labels))}")
    print(f"BM25 weighting: {ctfidf_model.bm25_weighting}")
    
    # Fit and transform
    ctfidf_matrix = ctfidf_model.fit_transform(doc_term_matrix, cluster_labels)
    
    print(f"c-TF-IDF matrix shape: {ctfidf_matrix.shape}")
    print(f"c-TF-IDF matrix type: {type(ctfidf_matrix)}")
    
    return ctfidf_matrix


def analyze_ctfidf_results(ctfidf_matrix: np.ndarray, 
                          feature_names: list, 
                          cluster_labels: list) -> None:
    """
    Analyzes and displays c-TF-IDF results.
    
    Args:
        ctfidf_matrix: c-TF-IDF weighted matrix.
        feature_names: List of feature (word) names.
        cluster_labels: Cluster assignments for each document.
    """
    unique_clusters = sorted(set(cluster_labels))
    n_clusters = len(unique_clusters)
    
    print(f"\nc-TF-IDF Analysis:")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Number of features: {ctfidf_matrix.shape[1]}")
    
    # Show top words for each cluster
    print(f"\nTop words per cluster:")
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip outliers
            continue
            
        cluster_idx = unique_clusters.index(cluster_id)
        cluster_scores = ctfidf_matrix[cluster_idx]
        
        # Get top words for this cluster
        top_word_indices = np.argsort(cluster_scores)[-5:][::-1]
        top_words = [feature_names[i] for i in top_word_indices]
        top_scores = [cluster_scores[i] for i in top_word_indices]
        
        print(f"  Cluster {cluster_id}:")
        for word, score in zip(top_words, top_scores):
            print(f"    {word}: {score:.4f}")


def compare_ctfidf_configurations(doc_term_matrix, cluster_labels: list, feature_names: list) -> None:
    """
    Compares different c-TF-IDF configurations.
    
    Args:
        doc_term_matrix: Document-term matrix.
        cluster_labels: Cluster assignments.
        feature_names: Feature names.
    """
    configurations = [
        {"bm25_weighting": False, "name": "Standard TF-IDF"},
        {"bm25_weighting": True, "name": "BM25 weighting"}
    ]
    
    print("\nComparing c-TF-IDF configurations:")
    for config in configurations:
        ctfidf_model = create_ctfidf_model(bm25_weighting=config["bm25_weighting"])
        ctfidf_matrix = demonstrate_ctfidf_transformer(ctfidf_model, doc_term_matrix, cluster_labels)
        
        # Calculate average scores
        avg_scores = np.mean(ctfidf_matrix, axis=0)
        max_scores = np.max(ctfidf_matrix, axis=0)
        
        print(f"  {config['name']}:")
        print(f"    Average scores: {np.mean(avg_scores):.4f}")
        print(f"    Max scores: {np.mean(max_scores):.4f}")


def demonstrate_topic_keywords(ctfidf_matrix: np.ndarray, 
                             feature_names: list, 
                             cluster_labels: list, 
                             n_words: int = 5) -> dict:
    """
    Demonstrates extracting topic keywords using c-TF-IDF scores.
    
    Args:
        ctfidf_matrix: c-TF-IDF weighted matrix.
        feature_names: List of feature names.
        cluster_labels: Cluster assignments.
        n_words: Number of top words to extract per topic.
    
    Returns:
        Dictionary mapping cluster IDs to top words.
    """
    unique_clusters = sorted(set(cluster_labels))
    topic_keywords = {}
    
    print(f"\nExtracting top {n_words} keywords per topic:")
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip outliers
            continue
            
        cluster_idx = unique_clusters.index(cluster_id)
        cluster_scores = ctfidf_matrix[cluster_idx]
        
        # Get top words
        top_word_indices = np.argsort(cluster_scores)[-n_words:][::-1]
        top_words = [(feature_names[i], cluster_scores[i]) for i in top_word_indices]
        
        topic_keywords[cluster_id] = top_words
        
        print(f"  Topic {cluster_id}: {[word for word, score in top_words]}")
    
    return topic_keywords


def main():
    """Demonstrate c-TF-IDF model creation and usage."""
    print("=== BERTopic c-TF-IDF Model Example ===\n")
    
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
    
    # Create document-term matrix first
    print("1. Creating document-term matrix...")
    vectorizer = CountVectorizer(max_df=0.8, stop_words="english")
    doc_term_matrix = vectorizer.fit_transform(sample_docs)
    feature_names = vectorizer.get_feature_names_out()
    print(f"   Document-term matrix shape: {doc_term_matrix.shape}")
    
    # Create mock cluster labels (simulating clustering results)
    cluster_labels = [0, 0, 1, 0, 2, 1, 0, 2, 0, 2, 1, 2, 0, 2, 1]
    print(f"   Cluster labels: {cluster_labels}")
    
    # Create default c-TF-IDF model
    print("\n2. Creating default c-TF-IDF model...")
    ctfidf_model = create_ctfidf_model()
    print(f"   c-TF-IDF model: {ctfidf_model}")
    print(f"   BM25 weighting: {ctfidf_model.bm25_weighting}")
    
    # Demonstrate transformation
    print("\n3. Applying c-TF-IDF transformation...")
    ctfidf_matrix = demonstrate_ctfidf_transformer(ctfidf_model, doc_term_matrix, cluster_labels)
    
    # Analyze results
    print("\n4. Analyzing c-TF-IDF results...")
    analyze_ctfidf_results(ctfidf_matrix, feature_names, cluster_labels)
    
    # Compare different configurations
    print("\n5. Comparing c-TF-IDF configurations...")
    compare_ctfidf_configurations(doc_term_matrix, cluster_labels, feature_names)
    
    # Demonstrate topic keyword extraction
    print("\n6. Extracting topic keywords...")
    topic_keywords = demonstrate_topic_keywords(ctfidf_matrix, feature_names, cluster_labels)
    
    # Show parameter effects
    print(f"\n7. c-TF-IDF parameter effects:")
    print(f"   bm25_weighting: True uses BM25 variant for better document length handling")
    print(f"   Standard TF-IDF: May be biased towards longer documents")
    print(f"   BM25 weighting: More robust to document length variations")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
