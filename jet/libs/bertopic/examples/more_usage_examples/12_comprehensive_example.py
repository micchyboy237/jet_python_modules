"""
BERTopic Usage Example: Comprehensive Example

This example demonstrates a complete BERTopic workflow using all the components 
and functions from the previous examples. It shows how to build a sophisticated 
topic modeling pipeline from start to finish.

Key Features:
- Complete BERTopic workflow
- All components working together
- Advanced topic modeling pipeline
- Topic analysis and visualization
- Export capabilities
- Best practices demonstration

Usage:
    python 12_comprehensive_example.py
"""

from typing import List, Optional
import umap
import hdbscan
from sklearn.feature_extraction.text import CountVectorizer
from jet.adapters.bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd


# Sample documents for testing (generic financial news snippets)
SAMPLE_DOCS: List[str] = [
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
    "Healthcare companies report positive results.",
    "Google announces new AI initiatives.",
    "Facebook rebrands to Meta platforms.",
    "Netflix reports subscriber growth.",
    "Uber expands food delivery services.",
    "Airbnb recovers from pandemic losses."
] * 5  # Repeat for ~100 docs to simulate dataset size


def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Creates a SentenceTransformer embedding model."""
    return SentenceTransformer(model_name)


def create_umap_model(n_neighbors: int = 5, min_dist: float = 0.01, random_state: int = 0) -> umap.UMAP:
    """Creates a UMAP model for dimensionality reduction."""
    return umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )


def create_hdbscan_model(min_cluster_size: int = 15, min_samples: int = 5, random_state: int = 0) -> hdbscan.HDBSCAN:
    """Creates an HDBSCAN model for clustering."""
    return hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
        cluster_selection_method='eom',
        min_samples=min_samples,
        random_state=random_state
    )


def create_vectorizer_model(max_df: float = 0.8, stop_words: Optional[str] = "english") -> CountVectorizer:
    """Creates a CountVectorizer for text vectorization."""
    return CountVectorizer(max_df=max_df, stop_words=stop_words)


def create_ctfidf_model(bm25_weighting: bool = True) -> ClassTfidfTransformer:
    """Creates a c-TF-IDF transformer for topic-level word weighting."""
    return ClassTfidfTransformer(bm25_weighting=bm25_weighting)


def create_keybert_representation_model() -> KeyBERTInspired:
    """Creates a KeyBERTInspired model for semantic keyword extraction."""
    return KeyBERTInspired()


def create_advanced_topic_model(
    embedding_model: SentenceTransformer,
    umap_model: umap.UMAP,
    documents: List[str],
    clustering_model: Optional[hdbscan.HDBSCAN] = None,
    vectorizer_model: Optional[CountVectorizer] = None,
    ctfidf_model: Optional[ClassTfidfTransformer] = None,
    representation_model: Optional[KeyBERTInspired] = None
) -> BERTopic:
    """Creates and fits an advanced BERTopic model with all components."""
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


def analyze_topic_results(topic_model: BERTopic, documents: List[str]) -> dict:
    """Comprehensive analysis of topic modeling results."""
    print("\n=== Comprehensive Topic Analysis ===")
    
    # Get topic assignments
    topics, probs = topic_model.transform(documents)
    
    # Basic statistics
    unique_topics = set(topics)
    n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
    n_outliers = np.sum(np.array(topics) == -1)
    
    print(f"Total documents: {len(documents)}")
    print(f"Number of topics: {n_topics}")
    print(f"Number of outliers: {n_outliers}")
    print(f"Outlier percentage: {n_outliers/len(documents)*100:.1f}%")
    
    # Topic information
    topic_info = topic_model.get_topic_info()
    print("\nTopic Information:")
    print(topic_info)
    
    # Topic keywords
    print("\nTopic Keywords:")
    for topic_id in unique_topics:
        if topic_id == -1:
            continue
        try:
            keywords = topic_model.get_topic(topic_id)
            if keywords:
                top_words = [word for word, score in keywords[:5]]
                print(f"Topic {topic_id}: {top_words}")
        except Exception as e:
            print(f"Topic {topic_id}: Error - {e}")
    
    # Topic document analysis
    print("\nTopic Document Analysis:")
    topic_documents = {}
    for doc, topic_id in zip(documents, topics):
        if topic_id not in topic_documents:
            topic_documents[topic_id] = []
        topic_documents[topic_id].append(doc)
    
    for topic_id in sorted(topic_documents.keys()):
        docs = topic_documents[topic_id]
        print(f"\nTopic {topic_id} ({len(docs)} documents):")
        for i, doc in enumerate(docs[:2]):  # Show first 2 docs
            print(f"  {i+1}. {doc}")
        if len(docs) > 2:
            print(f"  ... and {len(docs) - 2} more documents")
    
    return {
        'n_topics': n_topics,
        'n_outliers': n_outliers,
        'topic_info': topic_info,
        'topic_documents': topic_documents
    }


def demonstrate_workflow_variations(documents: List[str]) -> None:
    """Demonstrates different workflow variations."""
    print("\n=== Workflow Variations ===")
    
    # Variation 1: Basic pipeline
    print("\n1. Basic Pipeline (default components):")
    try:
        basic_model = BERTopic()
        topics_basic, probs_basic = basic_model.fit_transform(documents)
        n_topics_basic = len(set(topics_basic)) - (1 if -1 in topics_basic else 0)
        print(f"   Basic model: {n_topics_basic} topics")
    except Exception as e:
        print(f"   Basic model: Error - {e}")
    
    # Variation 2: Custom components
    print("\n2. Custom Components Pipeline:")
    try:
        emb_model = create_embedding_model()
        umap_model = create_umap_model()
        hdbscan_model = create_hdbscan_model()
        
        custom_model = BERTopic(
            embedding_model=emb_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model
        )
        topics_custom, probs_custom = custom_model.fit_transform(documents)
        n_topics_custom = len(set(topics_custom)) - (1 if -1 in topics_custom else 0)
        print(f"   Custom model: {n_topics_custom} topics")
    except Exception as e:
        print(f"   Custom model: Error - {e}")
    
    # Variation 3: Advanced pipeline
    print("\n3. Advanced Pipeline (all components):")
    try:
        emb_model = create_embedding_model()
        umap_model = create_umap_model()
        hdbscan_model = create_hdbscan_model()
        vectorizer_model = create_vectorizer_model()
        ctfidf_model = create_ctfidf_model()
        
        advanced_model = BERTopic(
            embedding_model=emb_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )
        topics_advanced, probs_advanced = advanced_model.fit_transform(documents)
        n_topics_advanced = len(set(topics_advanced)) - (1 if -1 in topics_advanced else 0)
        print(f"   Advanced model: {n_topics_advanced} topics")
    except Exception as e:
        print(f"   Advanced model: Error - {e}")


def export_comprehensive_analysis(topic_model: BERTopic, documents: List[str]) -> None:
    """Exports comprehensive topic analysis to files."""
    print("\n=== Exporting Analysis ===")
    
    # Get topic assignments
    topics, probs = topic_model.transform(documents)
    
    # Create comprehensive DataFrame
    analysis_data = []
    for i, (doc, topic_id, prob) in enumerate(zip(documents, topics, probs)):
        analysis_data.append({
            'document_id': i,
            'document': doc,
            'topic_id': topic_id,
            'topic_probability': prob
        })
    
    # Save to CSV
    df = pd.DataFrame(analysis_data)
    df.to_csv('comprehensive_topic_analysis.csv', index=False)
    print(f"   Exported {len(analysis_data)} document-topic assignments to comprehensive_topic_analysis.csv")
    
    # Export topic information
    topic_info = topic_model.get_topic_info()
    topic_info.to_csv('topic_information.csv', index=False)
    print("   Exported topic information to topic_information.csv")


def demonstrate_best_practices() -> None:
    """Demonstrates BERTopic best practices."""
    print("\n=== BERTopic Best Practices ===")
    
    print("1. Model Selection:")
    print("   - Use 'all-MiniLM-L6-v2' for lightweight, fast processing")
    print("   - Use 'BAAI/bge-base-en-v1.5' for better semantic understanding")
    print("   - Consider domain-specific models for specialized text")
    
    print("\n2. Parameter Tuning:")
    print("   - UMAP: Lower n_neighbors for local structure, higher for global")
    print("   - HDBSCAN: Adjust min_cluster_size based on expected topic size")
    print("   - Vectorizer: Use stop_words and max_df to filter noise")
    
    print("\n3. Workflow Recommendations:")
    print("   - Start with basic pipeline for quick prototyping")
    print("   - Add custom components for better results")
    print("   - Use advanced pipeline for production systems")
    print("   - Always validate results with domain experts")
    
    print("\n4. Performance Tips:")
    print("   - Pre-compute embeddings for large datasets")
    print("   - Use batch processing for document encoding")
    print("   - Consider model size vs. accuracy trade-offs")
    print("   - Monitor memory usage with large document sets")


def main():
    """Demonstrate comprehensive BERTopic workflow."""
    print("=== BERTopic Comprehensive Example ===\n")
    
    print(f"Sample dataset: {len(SAMPLE_DOCS)} documents")
    print("Sample documents preview:")
    for i, doc in enumerate(SAMPLE_DOCS[:3]):
        print(f"  {i+1}. {doc}")
    print(f"  ... and {len(SAMPLE_DOCS) - 3} more documents")
    
    # Create comprehensive pipeline
    print("\n=== Creating Comprehensive Pipeline ===")
    
    # Create all components
    embedding_model = create_embedding_model("all-MiniLM-L6-v2")
    umap_model = create_umap_model()
    hdbscan_model = create_hdbscan_model()
    vectorizer_model = create_vectorizer_model()
    ctfidf_model = create_ctfidf_model()
    
    print("Components created:")
    print(f"  - Embedding model: {embedding_model}")
    print(f"  - UMAP model: {umap_model}")
    print(f"  - HDBSCAN model: {hdbscan_model}")
    print(f"  - Vectorizer model: {vectorizer_model}")
    print(f"  - c-TF-IDF model: {ctfidf_model}")
    
    # Create advanced topic model
    print("\n=== Creating Advanced Topic Model ===")
    try:
        topic_model = create_advanced_topic_model(
            embedding_model=embedding_model,
            umap_model=umap_model,
            documents=SAMPLE_DOCS,
            clustering_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model
        )
        print("Advanced topic model created and fitted successfully!")
        
        # Analyze results
        analyze_topic_results(topic_model, SAMPLE_DOCS)
        
        # Demonstrate workflow variations
        demonstrate_workflow_variations(SAMPLE_DOCS)
        
        # Export analysis
        export_comprehensive_analysis(topic_model, SAMPLE_DOCS)
        
        # Show best practices
        demonstrate_best_practices()
        
    except Exception as e:
        print(f"Error creating advanced model: {e}")
        print("Falling back to basic model...")
        
        # Fallback to basic model
        basic_model = BERTopic()
        topics, probs = basic_model.fit_transform(SAMPLE_DOCS)
        print("Basic model created and fitted successfully!")
        
        # Analyze basic results
        analyze_topic_results(basic_model, SAMPLE_DOCS)
    
    print("\n=== Comprehensive Example Completed Successfully! ===")
    print("\nNext Steps:")
    print("1. Review the generated CSV files for detailed analysis")
    print("2. Experiment with different parameter configurations")
    print("3. Try different embedding models for your specific domain")
    print("4. Consider visualization with topic_model.visualize_topics()")
    print("5. Integrate with your specific use case and data")


if __name__ == "__main__":
    main()
