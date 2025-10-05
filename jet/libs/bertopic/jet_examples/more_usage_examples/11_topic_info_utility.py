"""
BERTopic Usage Example: Get Topic Information Utility

This example demonstrates how to extract and analyze topic information from a 
fitted BERTopic model. This utility function helps with topic inspection, 
analysis, and understanding the results of topic modeling.

Key Features:
- Topic information extraction
- Topic keyword analysis
- Topic document analysis
- Topic statistics and insights
- Integration with BERTopic models

Usage:
    python 11_topic_info_utility.py
"""

from typing import List, Optional
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
import numpy as np
import pandas as pd


def get_topic_info(topic_model: BERTopic) -> None:
    """
    Retrieves and prints topic information (ID, count, top words).
    
    Args:
        topic_model: Fitted BERTopic model.
    
    Returns:
        None (prints to console for quick inspection).
    
    Example:
        info = get_topic_info(model)
    """
    info = topic_model.get_topic_info()
    print("Topic Information:")
    print(info)


def analyze_topic_keywords(topic_model: BERTopic, n_words: int = 10) -> dict:
    """
    Analyzes and displays topic keywords for all topics.
    
    Args:
        topic_model: Fitted BERTopic model.
        n_words: Number of top words to show per topic.
    
    Returns:
        Dictionary mapping topic IDs to their keywords.
    """
    print(f"\nTopic Keywords (top {n_words} words per topic):")
    
    topic_info = topic_model.get_topic_info()
    topic_keywords = {}
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:  # Skip outliers
            continue
            
        try:
            keywords = topic_model.get_topic(topic_id)
            if keywords:
                top_words = keywords[:n_words]
                topic_keywords[topic_id] = top_words
                
                print(f"\nTopic {topic_id} ({row['Count']} documents):")
                for word, score in top_words:
                    print(f"  {word}: {score:.4f}")
            else:
                print(f"\nTopic {topic_id}: No keywords available")
                
        except Exception as e:
            print(f"\nTopic {topic_id}: Error getting keywords - {e}")
    
    return topic_keywords


def analyze_topic_documents(topic_model: BERTopic, documents: List[str]) -> dict:
    """
    Analyzes documents assigned to each topic.
    
    Args:
        topic_model: Fitted BERTopic model.
        documents: List of original documents.
    
    Returns:
        Dictionary mapping topic IDs to their documents.
    """
    print(f"\nTopic Document Analysis:")
    
    # Get topic assignments
    topics, probs = topic_model.transform(documents)
    
    # Group documents by topic
    topic_documents = {}
    for doc, topic_id in zip(documents, topics):
        if topic_id not in topic_documents:
            topic_documents[topic_id] = []
        topic_documents[topic_id].append(doc)
    
    # Display documents per topic
    for topic_id in sorted(topic_documents.keys()):
        docs = topic_documents[topic_id]
        print(f"\nTopic {topic_id} ({len(docs)} documents):")
        for i, doc in enumerate(docs[:3]):  # Show first 3 documents
            print(f"  {i+1}. {doc}")
        if len(docs) > 3:
            print(f"  ... and {len(docs) - 3} more documents")
    
    return topic_documents


def calculate_topic_statistics(topic_model: BERTopic, documents: List[str]) -> dict:
    """
    Calculates and displays topic statistics.
    
    Args:
        topic_model: Fitted BERTopic model.
        documents: List of original documents.
    
    Returns:
        Dictionary of topic statistics.
    """
    print(f"\nTopic Statistics:")
    
    # Get topic assignments
    topics, probs = topic_model.transform(documents)
    
    # Calculate statistics
    unique_topics = set(topics)
    n_topics = len(unique_topics) - (1 if -1 in unique_topics else 0)
    n_outliers = np.sum(np.array(topics) == -1)
    n_documents = len(documents)
    
    # Topic size distribution
    topic_sizes = {}
    for topic_id in unique_topics:
        topic_sizes[topic_id] = np.sum(np.array(topics) == topic_id)
    
    # Average topic probability
    avg_prob = np.mean(probs) if probs is not None else 0
    
    # Display statistics
    print(f"  Total documents: {n_documents}")
    print(f"  Number of topics: {n_topics}")
    print(f"  Number of outliers: {n_outliers}")
    print(f"  Outlier percentage: {n_outliers/n_documents*100:.1f}%")
    print(f"  Average topic probability: {avg_prob:.4f}")
    
    print(f"\n  Topic size distribution:")
    for topic_id in sorted(topic_sizes.keys()):
        size = topic_sizes[topic_id]
        percentage = size / n_documents * 100
        if topic_id == -1:
            print(f"    Outliers: {size} documents ({percentage:.1f}%)")
        else:
            print(f"    Topic {topic_id}: {size} documents ({percentage:.1f}%)")
    
    return {
        'n_topics': n_topics,
        'n_outliers': n_outliers,
        'n_documents': n_documents,
        'topic_sizes': topic_sizes,
        'avg_probability': avg_prob
    }


def find_similar_topics(topic_model: BERTopic, topic_id: int, n_similar: int = 3) -> List[tuple]:
    """
    Finds topics similar to a given topic.
    
    Args:
        topic_model: Fitted BERTopic model.
        topic_id: ID of the topic to find similar topics for.
        n_similar: Number of similar topics to return.
    
    Returns:
        List of (similar_topic_id, similarity_score) tuples.
    """
    print(f"\nFinding topics similar to Topic {topic_id}:")
    
    try:
        # Get topic representations
        topic_representations = topic_model.get_topics()
        
        if topic_id not in topic_representations:
            print(f"  Topic {topic_id} not found")
            return []
        
        # Calculate similarities (simplified approach)
        similarities = []
        for other_topic_id, other_representation in topic_representations.items():
            if other_topic_id == topic_id or other_topic_id == -1:
                continue
                
            # Simple similarity based on shared keywords
            topic_words = set([word for word, score in topic_representations[topic_id]])
            other_words = set([word for word, score in other_representation])
            
            if len(topic_words) > 0 and len(other_words) > 0:
                similarity = len(topic_words.intersection(other_words)) / len(topic_words.union(other_words))
                similarities.append((other_topic_id, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Display results
        for i, (similar_topic_id, similarity) in enumerate(similarities[:n_similar]):
            print(f"  {i+1}. Topic {similar_topic_id}: {similarity:.4f}")
        
        return similarities[:n_similar]
        
    except Exception as e:
        print(f"  Error finding similar topics: {e}")
        return []


def export_topic_analysis(topic_model: BERTopic, documents: List[str], filename: str = "topic_analysis.csv") -> None:
    """
    Exports topic analysis to a CSV file.
    
    Args:
        topic_model: Fitted BERTopic model.
        documents: List of original documents.
        filename: Output filename for the CSV file.
    """
    print(f"\nExporting topic analysis to {filename}...")
    
    # Get topic assignments
    topics, probs = topic_model.transform(documents)
    
    # Create analysis DataFrame
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
    df.to_csv(filename, index=False)
    
    print(f"  Exported {len(analysis_data)} document-topic assignments")
    print(f"  File saved as: {filename}")


def main():
    """Demonstrate topic information utility functions."""
    print("=== BERTopic Topic Information Utility Example ===\n")
    
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
    
    # Create a fitted BERTopic model
    print("1. Creating and fitting BERTopic model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.01, random_state=0)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=3, min_samples=2)
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model
    )
    
    topics, probs = topic_model.fit_transform(sample_docs)
    print(f"   Model fitted with {len(set(topics)) - (1 if -1 in topics else 0)} topics")
    
    # Demonstrate topic information extraction
    print("\n2. Extracting topic information...")
    get_topic_info(topic_model)
    
    # Analyze topic keywords
    print("\n3. Analyzing topic keywords...")
    topic_keywords = analyze_topic_keywords(topic_model)
    
    # Analyze topic documents
    print("\n4. Analyzing topic documents...")
    topic_documents = analyze_topic_documents(topic_model, sample_docs)
    
    # Calculate topic statistics
    print("\n5. Calculating topic statistics...")
    topic_stats = calculate_topic_statistics(topic_model, sample_docs)
    
    # Find similar topics
    print("\n6. Finding similar topics...")
    if len(topic_keywords) > 1:
        first_topic = list(topic_keywords.keys())[0]
        similar_topics = find_similar_topics(topic_model, first_topic)
    
    # Export analysis
    print("\n7. Exporting topic analysis...")
    export_topic_analysis(topic_model, sample_docs)
    
    # Show utility benefits
    print(f"\n8. Topic information utility benefits:")
    print(f"   - Quick topic inspection and analysis")
    print(f"   - Topic keyword and document analysis")
    print(f"   - Topic statistics and insights")
    print(f"   - Similar topic discovery")
    print(f"   - Export capabilities for further analysis")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
