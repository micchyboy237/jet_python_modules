"""
BERTopic Usage Example: Create TfidfVectorizer Model
This example demonstrates how to configure a TfidfVectorizer for text vectorization 
in BERTopic. The vectorizer creates document-term matrices with TF-IDF weighting 
for topic modeling and keyword extraction.
Key Features:
- TF-IDF based text preprocessing
- Stop word removal
- N-gram support
- Document frequency filtering
- Integration with BERTopic pipeline
Usage:
    python 08_tfidf_vectorizer.py
"""
from typing import Optional
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from jet.adapters.bertopic import BERTopic

def create_tfidf_vectorizer(max_df: float = 0.8, stop_words: Optional[str] = "english") -> TfidfVectorizer:
    """
    Creates a TfidfVectorizer for document-term matrix creation with TF-IDF weighting.
    Args:
        max_df: Ignore words appearing in > max_df fraction of docs (reduces noise).
        stop_words: Stop words to remove (e.g., "english" or None).
    Returns:
        Initialized TfidfVectorizer instance.
    Example:
        vectorizer = create_tfidf_vectorizer(max_df=0.8, stop_words="english")
    """
    return TfidfVectorizer(max_df=max_df, stop_words=stop_words)

def demonstrate_tfidf_vectorizer(vectorizer: TfidfVectorizer, documents: list) -> tuple:
    """
    Demonstrates TfidfVectorizer on a set of documents.
    Args:
        vectorizer: Configured TfidfVectorizer model.
        documents: List of documents to vectorize.
    Returns:
        Tuple of (document-term matrix, feature names).
    """
    print(f"Input documents: {len(documents)}")
    doc_term_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Document-term matrix shape: {doc_term_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    return doc_term_matrix, feature_names

def analyze_tfidf_results(doc_term_matrix, feature_names: list, documents: list) -> None:
    """
    Analyzes and displays TfidfVectorizer results.
    Args:
        doc_term_matrix: Sparse matrix of document-term TF-IDF weights.
        feature_names: List of feature (word) names.
        documents: Original documents.
    """
    print("\nTfidfVectorizer Analysis:")
    print(f"  Total documents: {doc_term_matrix.shape[0]}")
    print(f"  Total features: {doc_term_matrix.shape[1]}")
    print(f"  Average TF-IDF score per document: {doc_term_matrix.sum() / doc_term_matrix.shape[0]:.3f}")
    word_scores = doc_term_matrix.sum(axis=0).A1
    word_score_pairs = list(zip(feature_names, word_scores))
    word_score_pairs.sort(key=lambda x: x[1], reverse=True)
    print("\nTop words by TF-IDF score:")
    for word, score in word_score_pairs[:10]:
        print(f"  {word}: {score:.3f}")
    doc_scores = doc_term_matrix.sum(axis=1).A1
    max_score_idx = np.argmax(doc_scores)
    print("\nDocument with highest TF-IDF score:")
    print(f"  Index: {max_score_idx}")
    print(f"  Score: {doc_scores[max_score_idx]:.3f}")
    print(f"  Text: {documents[max_score_idx]}")

def demonstrate_bertopic_integration(documents: list, vectorizer: TfidfVectorizer) -> None:
    """
    Demonstrates integration of TfidfVectorizer with BERTopic.
    Args:
        documents: List of documents to process.
        vectorizer: Configured TfidfVectorizer model.
    """
    print("\nDemonstrating TfidfVectorizer with BERTopic:")
    topic_model = BERTopic(vectorizer_model=vectorizer, min_topic_size=2)
    topics, probs = topic_model.fit_transform(documents)
    print(f"  Number of topics found: {len(set(topics)) - (1 if -1 in topics else 0)}")
    topic_info = topic_model.get_topic_info()
    print("\nTopic Information:")
    for _, row in topic_info.iterrows():
        if row['Topic'] != -1:
            print(f"  Topic {row['Topic']}: {row['Name']} (Count: {row['Count']})")
    print("\nTop words for each topic:")
    for topic in set(topics):
        if topic != -1:
            words = topic_model.get_topic(topic)
            print(f"  Topic {topic}: {[(word, round(score, 3)) for word, score in words[:5]]}")

def compare_tfidf_configurations(documents: list) -> None:
    """
    Compares different TfidfVectorizer configurations.
    Args:
        documents: List of documents to test.
    """
    configurations = [
        {"max_df": 1.0, "stop_words": None, "name": "No filtering"},
        {"max_df": 0.8, "stop_words": "english", "name": "Default"},
        {"max_df": 0.5, "stop_words": "english", "name": "Strict filtering"},
        {"max_df": 0.9, "stop_words": None, "name": "Minimal filtering"}
    ]
    print("\nComparing TfidfVectorizer configurations:")
    for config in configurations:
        vectorizer = create_tfidf_vectorizer(
            max_df=config["max_df"],
            stop_words=config["stop_words"]
        )
        doc_term_matrix, feature_names = demonstrate_tfidf_vectorizer(vectorizer, documents)
        print(f"  {config['name']}: {doc_term_matrix.shape[1]} features")

def demonstrate_ngrams(documents: list) -> None:
    """
    Demonstrates n-gram extraction with TfidfVectorizer.
    Args:
        documents: List of documents to test.
    """
    print("\nDemonstrating n-gram extraction:")
    vectorizer_1gram = TfidfVectorizer(ngram_range=(1, 1), max_df=0.8, stop_words="english")
    doc_term_1gram, features_1gram = demonstrate_tfidf_vectorizer(vectorizer_1gram, documents)
    vectorizer_2gram = TfidfVectorizer(ngram_range=(2, 2), max_df=0.8, stop_words="english")
    doc_term_2gram, features_2gram = demonstrate_tfidf_vectorizer(vectorizer_2gram, documents)
    vectorizer_12gram = TfidfVectorizer(ngram_range=(1, 2), max_df=0.8, stop_words="english")
    doc_term_12gram, features_12gram = demonstrate_tfidf_vectorizer(vectorizer_12gram, documents)
    print(f"  1-grams: {len(features_1gram)} features")
    print(f"  2-grams: {len(features_2gram)} features")
    print(f"  1-2 grams: {len(features_12gram)} features")
    bigrams = [f for f in features_2gram if ' ' in f][:5]
    print(f"  Sample bigrams: {bigrams}")

def main():
    """Demonstrate TfidfVectorizer model creation and usage with BERTopic."""
    print("=== BERTopic TfidfVectorizer Model Example ===\n")
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
    print("1. Creating default TfidfVectorizer...")
    vectorizer = create_tfidf_vectorizer()
    print(f"   Vectorizer: {vectorizer}")
    print(f"   max_df: {vectorizer.max_df}")
    print(f"   stop_words: {vectorizer.stop_words}")
    print("\n2. Applying TfidfVectorizer...")
    doc_term_matrix, feature_names = demonstrate_tfidf_vectorizer(vectorizer, sample_docs)
    print("\n3. Analyzing TfidfVectorizer results...")
    analyze_tfidf_results(doc_term_matrix, feature_names, sample_docs)
    print("\n4. Demonstrating BERTopic integration...")
    demonstrate_bertopic_integration(sample_docs, vectorizer)
    print("\n5. Comparing TfidfVectorizer configurations...")
    compare_tfidf_configurations(sample_docs)
    print("\n6. Demonstrating n-gram extraction...")
    demonstrate_ngrams(sample_docs)
    print("\n7. TfidfVectorizer parameter effects:")
    print("   max_df: Higher values keep more common words")
    print("   stop_words: Removes common words like 'the', 'and', etc.")
    print("   ngram_range: (1,1) for unigrams, (1,2) for unigrams+bigrams")
    print("   min_df: Minimum document frequency for words")
    print("\n=== Example completed successfully! ===")

if __name__ == "__main__":
    main()
