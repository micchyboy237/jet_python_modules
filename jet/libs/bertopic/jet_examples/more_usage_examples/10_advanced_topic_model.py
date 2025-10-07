"""
BERTopic Usage Example: Create Advanced BERTopic Model

This example demonstrates how to create and fit an advanced BERTopic model with 
all components (vectorizer, c-TF-IDF, KeyBERT) for refined topic modeling. 
This provides the most comprehensive topic modeling pipeline with all available refinements.

Key Features:
- Full pipeline with all components
- Custom vectorizer for text processing
- c-TF-IDF for topic-level word weighting
- KeyBERT for semantic keyword extraction
- Advanced topic modeling capabilities

Usage:
    python 10_advanced_topic_model.py
"""

from typing import List, Optional
import umap
import hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from jet.adapters.bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np


def create_advanced_topic_model(
    embedding_model: SentenceTransformer,
    umap_model: umap.UMAP,
    documents: List[str],
    clustering_model: Optional[hdbscan.HDBSCAN] = None,  # Or KMeans
    vectorizer_model: Optional[CountVectorizer] = None,
    ctfidf_model: Optional[ClassTfidfTransformer] = None,
    representation_model: Optional[KeyBERTInspired] = None
) -> BERTopic:
    """
    Creates and fits an advanced BERTopic model with optional refinements.
    
    Args:
        embedding_model: Required embedding model.
        umap_model: Required UMAP model.
        clustering_model: Optional clustering (HDBSCAN or KMeans).
        vectorizer_model: Optional vectorizer for text matrix.
        ctfidf_model: Optional c-TF-IDF for word weighting.
        representation_model: Optional KeyBERT for semantic keywords.
        documents: List of input documents.
    
    Returns:
        Fitted BERTopic model.
    
    Example:
        emb = create_embedding_model("BAAI/bge-base-en-v1.5")
        um = create_umap_model()
        vec = create_vectorizer_model()
        hdb = create_hdbscan_model()
        keybert = create_keybert_representation_model()
        model = create_advanced_topic_model(
            embedding_model=emb, umap_model=um, clustering_model=hdb,
            vectorizer_model=vec, representation_model=keybert, documents=SAMPLE_DOCS
        )
        topic_info = model.get_topic_info()
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=clustering_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model
    )
    
    topic_model.fit_transform(documents)
    return topic_model


def demonstrate_full_pipeline(documents: List[str]) -> BERTopic:
    """
    Demonstrates the full advanced BERTopic pipeline.
    
    Args:
        documents: List of documents to model.
    
    Returns:
        Fitted BERTopic model.
    """
    print("1. Creating full advanced pipeline...")
    
    # Create all components
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    vectorizer_model = CountVectorizer(max_df=0.8, stop_words="english")
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    print(f"   Embedding model: {embedding_model}")
    print(f"   UMAP model: {umap_model}")
    print(f"   HDBSCAN model: {hdbscan_model}")
    print(f"   Vectorizer model: {vectorizer_model}")
    print(f"   c-TF-IDF model: {ctfidf_model}")
    
    # Create advanced model
    topic_model = create_advanced_topic_model(
        embedding_model=embedding_model,
        umap_model=umap_model,
        documents=documents,
        clustering_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )
    
    print("   Advanced model created and fitted")
    return topic_model


def demonstrate_with_keybert(documents: List[str]) -> BERTopic:
    """
    Demonstrates advanced BERTopic with KeyBERT representation.
    
    Args:
        documents: List of documents to model.
    
    Returns:
        Fitted BERTopic model.
    """
    print("\n2. Creating advanced pipeline with KeyBERT...")
    
    # Create components including KeyBERT
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    vectorizer_model = CountVectorizer(max_df=0.8, stop_words="english")
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    try:
        keybert_model = KeyBERTInspired()
        print(f"   KeyBERT model: {keybert_model}")
        
        # Create advanced model with KeyBERT
        topic_model = create_advanced_topic_model(
            embedding_model=embedding_model,
            umap_model=umap_model,
            documents=documents,
            clustering_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=keybert_model
        )
        
        print("   Advanced model with KeyBERT created and fitted")
        return topic_model
        
    except Exception as e:
        print(f"   KeyBERT not available: {e}")
        print("   Creating model without KeyBERT...")
        
        # Create without KeyBERT
        topic_model = create_advanced_topic_model(
            embedding_model=embedding_model,
            umap_model=umap_model,
            documents=documents,
            clustering_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )
        
        print("   Advanced model without KeyBERT created and fitted")
        return topic_model


def demonstrate_kmeans_advanced(documents: List[str]) -> BERTopic:
    """
    Demonstrates advanced BERTopic with K-Means clustering.
    
    Args:
        documents: List of documents to model.
    
    Returns:
        Fitted BERTopic model.
    """
    print("\n3. Creating advanced pipeline with K-Means...")
    
    # Create components with K-Means
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    kmeans_model = KMeans(n_clusters=5, random_state=0)
    vectorizer_model = CountVectorizer(max_df=0.8, stop_words="english")
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
    
    print(f"   K-Means model: {kmeans_model}")
    
    # Create advanced model with K-Means
    topic_model = create_advanced_topic_model(
        embedding_model=embedding_model,
        umap_model=umap_model,
        documents=documents,
        clustering_model=kmeans_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model
    )
    
    print("   Advanced model with K-Means created and fitted")
    return topic_model


def analyze_advanced_results(topic_model: BERTopic, documents: List[str]) -> None:
    """
    Analyzes and displays advanced topic modeling results.
    
    Args:
        topic_model: Fitted advanced BERTopic model.
        documents: Original documents.
    """
    print("\n4. Analyzing advanced results...")
    
    # Get topic information
    topic_info = topic_model.get_topic_info()
    print("   Topic information:")
    print(topic_info)
    
    # Get topic assignments
    try:
        topics, probs = topic_model.transform(documents)
    except AttributeError as e:
        if "No prediction data was generated" in str(e):
            print(f"   Warning: {e}")
            print("   Using fit_transform instead of transform...")
            topics, probs = topic_model.fit_transform(documents)
        else:
            raise e
    print(f"\n   Topic assignments: {topics}")
    print(f"   Topic probabilities shape: {probs.shape}")
    
    # Show topic keywords
    print("\n   Topic keywords:")
    for topic_id in topic_info['Topic'].values:
        if topic_id == -1:
            continue
        try:
            keywords = topic_model.get_topic(topic_id)
            if keywords:
                top_words = [word for word, score in keywords[:5]]
                print(f"   Topic {topic_id}: {top_words}")
        except Exception as e:
            print(f"   Topic {topic_id}: Error getting keywords - {e}")


def compare_advanced_configurations(documents: List[str]) -> None:
    """
    Compares different advanced BERTopic configurations.
    
    Args:
        documents: List of documents to test.
    """
    print("\n5. Comparing advanced configurations...")
    
    # Basic advanced (no KeyBERT)
    try:
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
        hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
        vectorizer_model = CountVectorizer(max_df=0.8, stop_words="english")
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True)
        
        topic_model = create_advanced_topic_model(
            embedding_model=embedding_model,
            umap_model=umap_model,
            documents=documents,
            clustering_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )
        
        topics, probs = topic_model.transform(documents)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = np.sum(np.array(topics) == -1)
        print(f"   Advanced (no KeyBERT): {n_topics} topics, {n_outliers} outliers")
        
    except Exception as e:
        print(f"   Advanced (no KeyBERT): Error - {e}")
    
    # With KeyBERT
    try:
        keybert_model = KeyBERTInspired()
        topic_model_keybert = create_advanced_topic_model(
            embedding_model=embedding_model,
            umap_model=umap_model,
            clustering_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=keybert_model,
            documents=documents
        )
        
        topics_keybert, probs_keybert = topic_model_keybert.transform(documents)
        n_topics_keybert = len(set(topics_keybert)) - (1 if -1 in topics_keybert else 0)
        n_outliers_keybert = np.sum(np.array(topics_keybert) == -1)
        print(f"   Advanced (with KeyBERT): {n_topics_keybert} topics, {n_outliers_keybert} outliers")
        
    except Exception as e:
        print(f"   Advanced (with KeyBERT): Error - {e}")


def main():
    """Demonstrate advanced BERTopic model creation and usage."""
    print("=== BERTopic Advanced Model Example ===\n")
    
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
    
    # Demonstrate full pipeline
    topic_model = demonstrate_full_pipeline(sample_docs)
    
    # Demonstrate with KeyBERT
    topic_model_keybert = demonstrate_with_keybert(sample_docs)
    
    # Demonstrate with K-Means
    topic_model_kmeans = demonstrate_kmeans_advanced(sample_docs)
    
    # Analyze results
    analyze_advanced_results(topic_model, sample_docs)
    
    # Compare configurations
    compare_advanced_configurations(sample_docs)
    
    # Show advanced features
    print("\n6. Advanced BERTopic features:")
    print("   - Custom vectorizer for better text processing")
    print("   - c-TF-IDF for improved topic keywords")
    print("   - KeyBERT for semantic keyword extraction")
    print("   - Multiple clustering algorithms")
    print("   - Topic visualization and analysis")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
