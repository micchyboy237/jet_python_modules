"""
BERTopic Usage Example: Create CountVectorizer Model

This example demonstrates how to configure a CountVectorizer for text vectorization 
in BERTopic. The vectorizer creates document-term matrices that are used for topic 
refinement and keyword extraction.

Key Features:
- Configurable text preprocessing
- Stop word removal
- N-gram support
- Document frequency filtering
- Integration with BERTopic pipeline

Usage:
    python 06_vectorizer_model.py
"""

from typing import Optional
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def create_vectorizer_model(max_df: float = 0.8, stop_words: Optional[str] = "english") -> CountVectorizer:
    """
    Creates a CountVectorizer for document-term matrix creation.
    
    Args:
        max_df: Ignore words appearing in > max_df fraction of docs (reduces noise).
        stop_words: Stop words to remove (e.g., "english" or None).
    
    Returns:
        Initialized CountVectorizer instance.
    
    Example:
        vectorizer = create_vectorizer_model(max_df=0.8, stop_words="english")
    """
    return CountVectorizer(max_df=max_df, stop_words=stop_words)


def demonstrate_vectorizer(vectorizer: CountVectorizer, documents: list) -> tuple:
    """
    Demonstrates CountVectorizer on a set of documents.
    
    Args:
        vectorizer: Configured CountVectorizer model.
        documents: List of documents to vectorize.
    
    Returns:
        Tuple of (document-term matrix, feature names).
    """
    print(f"Input documents: {len(documents)}")
    
    # Fit and transform documents
    doc_term_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return doc_term_matrix, feature_names


def analyze_vectorizer_results(doc_term_matrix, feature_names: list, documents: list) -> None:
    """
    Analyzes and displays vectorizer results.
    
    Args:
        doc_term_matrix: Sparse matrix of document-term counts.
        feature_names: List of feature (word) names.
        documents: Original documents.
    """
    print(f"\nVectorizer Analysis:")
    print(f"  Total documents: {doc_term_matrix.shape[0]}")
    print(f"  Total features: {doc_term_matrix.shape[1]}")
    print(f"  Average features per document: {doc_term_matrix.sum() / doc_term_matrix.shape[0]:.1f}")
    
    # Show most common words
    word_counts = doc_term_matrix.sum(axis=0).A1
    word_freq_pairs = list(zip(feature_names, word_counts))
    word_freq_pairs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost common words:")
    for word, count in word_freq_pairs[:10]:
        print(f"  {word}: {count} occurrences")
    
    # Show document with most features
    doc_lengths = doc_term_matrix.sum(axis=1).A1
    max_length_idx = np.argmax(doc_lengths)
    print(f"\nDocument with most features:")
    print(f"  Index: {max_length_idx}")
    print(f"  Features: {doc_lengths[max_length_idx]}")
    print(f"  Text: {documents[max_length_idx]}")


def compare_vectorizer_configurations(documents: list) -> None:
    """
    Compares different CountVectorizer configurations.
    
    Args:
        documents: List of documents to test.
    """
    configurations = [
        {"max_df": 1.0, "stop_words": None, "name": "No filtering"},
        {"max_df": 0.8, "stop_words": "english", "name": "Default"},
        {"max_df": 0.5, "stop_words": "english", "name": "Strict filtering"},
        {"max_df": 0.9, "stop_words": None, "name": "Minimal filtering"}
    ]
    
    print("\nComparing CountVectorizer configurations:")
    for config in configurations:
        vectorizer = create_vectorizer_model(
            max_df=config["max_df"],
            stop_words=config["stop_words"]
        )
        doc_term_matrix, feature_names = demonstrate_vectorizer(vectorizer, documents)
        
        print(f"  {config['name']}: {doc_term_matrix.shape[1]} features")


def demonstrate_ngrams(documents: list) -> None:
    """
    Demonstrates n-gram extraction with CountVectorizer.
    
    Args:
        documents: List of documents to test.
    """
    print("\nDemonstrating n-gram extraction:")
    
    # Unigrams only
    vectorizer_1gram = CountVectorizer(ngram_range=(1, 1), max_df=0.8, stop_words="english")
    doc_term_1gram, features_1gram = demonstrate_vectorizer(vectorizer_1gram, documents)
    
    # Bigrams only
    vectorizer_2gram = CountVectorizer(ngram_range=(2, 2), max_df=0.8, stop_words="english")
    doc_term_2gram, features_2gram = demonstrate_vectorizer(vectorizer_2gram, documents)
    
    # Unigrams and bigrams
    vectorizer_12gram = CountVectorizer(ngram_range=(1, 2), max_df=0.8, stop_words="english")
    doc_term_12gram, features_12gram = demonstrate_vectorizer(vectorizer_12gram, documents)
    
    print(f"  1-grams: {len(features_1gram)} features")
    print(f"  2-grams: {len(features_2gram)} features")
    print(f"  1-2 grams: {len(features_12gram)} features")
    
    # Show some bigrams
    bigrams = [f for f in features_2gram if ' ' in f][:5]
    print(f"  Sample bigrams: {bigrams}")


def main():
    """Demonstrate CountVectorizer model creation and usage."""
    print("=== BERTopic CountVectorizer Model Example ===\n")
    
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
    
    # Create default vectorizer
    print("1. Creating default CountVectorizer...")
    vectorizer = create_vectorizer_model()
    print(f"   Vectorizer: {vectorizer}")
    print(f"   max_df: {vectorizer.max_df}")
    print(f"   stop_words: {vectorizer.stop_words}")
    
    # Demonstrate vectorization
    print("\n2. Applying CountVectorizer...")
    doc_term_matrix, feature_names = demonstrate_vectorizer(vectorizer, sample_docs)
    
    # Analyze results
    print("\n3. Analyzing vectorizer results...")
    analyze_vectorizer_results(doc_term_matrix, feature_names, sample_docs)
    
    # Compare different configurations
    print("\n4. Comparing vectorizer configurations...")
    compare_vectorizer_configurations(sample_docs)
    
    # Demonstrate n-grams
    print("\n5. Demonstrating n-gram extraction...")
    demonstrate_ngrams(sample_docs)
    
    # Show parameter effects
    print(f"\n6. CountVectorizer parameter effects:")
    print(f"   max_df: Higher values keep more common words")
    print(f"   stop_words: Removes common words like 'the', 'and', etc.")
    print(f"   ngram_range: (1,1) for unigrams, (1,2) for unigrams+bigrams")
    print(f"   min_df: Minimum document frequency for words")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
