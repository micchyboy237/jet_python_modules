from jet.adapters.bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Optional, List, Tuple

from jet.libs.bertopic.jet_examples.mock import load_sample_data
from jet.file.utils import save_file
import os
import shutil

from jet.logger import logger
from jet.transformers.formatters import format_json

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def update_topic_representation(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    n_gram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    vectorizer: CountVectorizer | None = None,
    min_df: int = 2,
    max_df: float = 0.8
) -> BERTopic:
    """
    Update the representation (keywords) of topics using a custom vectorizer or n-gram range.
    
    Args:
        topic_model: The fitted BERTopic model
        docs: List of documents
        topics: List of topic assignments
        n_gram_range: Range of n-grams to extract
        stop_words: Language for stop words or None to disable
        vectorizer: Custom vectorizer (if None, creates CountVectorizer)
        min_df: Minimum document frequency for terms
        max_df: Maximum document frequency for terms
        
    Returns:
        Updated BERTopic model
    """
    if vectorizer is None:
        vectorizer = CountVectorizer(
            stop_words=stop_words, 
            ngram_range=n_gram_range,
            min_df=min_df,
            max_df=max_df
        )
    
    topic_model.update_topics(docs, topics, vectorizer_model=vectorizer)
    return topic_model


def create_custom_vectorizer(
    vectorizer_type: str = "count",
    n_gram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    min_df: int = 2,
    max_df: float = 0.8,
    max_features: Optional[int] = None
) -> CountVectorizer | TfidfVectorizer:
    """
    Create a custom vectorizer for topic representation.
    
    Args:
        vectorizer_type: Type of vectorizer ("count" or "tfidf")
        n_gram_range: Range of n-grams to extract
        stop_words: Language for stop words
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        max_features: Maximum number of features
        
    Returns:
        Configured vectorizer
    """
    common_params = {
        "ngram_range": n_gram_range,
        "stop_words": stop_words,
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features
    }
    
    if vectorizer_type.lower() == "tfidf":
        return TfidfVectorizer(**common_params)
    else:
        return CountVectorizer(**common_params)


def compare_topic_representations(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    original_topics: dict = None
) -> dict:
    """
    Compare topic representations before and after updating.
    
    Args:
        topic_model: The BERTopic model
        docs: List of documents
        topics: List of topic assignments
        original_topics: Original topic representations (optional)
        
    Returns:
        dict: Comparison results
    """
    current_topics = {}
    for topic_id in range(len(topic_model.get_topic_info())):
        current_topics[topic_id] = topic_model.get_topic(topic_id)
    
    comparison = {
        "current_topics": current_topics,
        "n_topics": len(current_topics)
    }
    
    if original_topics:
        comparison["original_topics"] = original_topics
        comparison["changed_topics"] = []
        
        for topic_id in current_topics:
            if topic_id in original_topics:
                if current_topics[topic_id] != original_topics[topic_id]:
                    comparison["changed_topics"].append(topic_id)
    
    return comparison


def get_topic_keywords(
    topic_model: BERTopic,
    topic_id: int,
    top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Get top keywords for a specific topic.
    
    Args:
        topic_model: The BERTopic model
        topic_id: ID of the topic
        top_n: Number of top keywords to return
        
    Returns:
        List of (word, score) tuples
    """
    topic_words = topic_model.get_topic(topic_id)
    return topic_words[:top_n]


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents
    # docs = [
    #     "Machine learning and artificial intelligence are revolutionizing technology.",
    #     "Data science involves statistics, programming, and domain expertise.",
    #     "COVID-19 pandemic has changed global health and economy.",
    #     "Vaccines and medical research are crucial for public health.",
    #     "Quantum computing could break current encryption methods.",
    #     "Cryptocurrency and blockchain technology are emerging trends.",
    #     "Climate change is affecting weather patterns worldwide.",
    #     "Renewable energy sources like solar and wind are growing.",
    #     "Stock market volatility affects investor confidence.",
    #     "Economic policies influence inflation and employment rates.",
    #     "Deep learning neural networks require large datasets.",
    #     "Natural language processing is advancing rapidly.",
    #     "Computer vision applications are expanding in healthcare.",
    #     "Robotics and automation are transforming manufacturing.",
    #     "Internet of Things devices are becoming more prevalent."
    # ]
    docs = load_sample_data()
    
    print("=== Initial Topic Model ===")
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Store original topics for comparison
    original_topics = {}
    for topic_id in range(len(model.get_topic_info())):
        original_topics[topic_id] = model.get_topic(topic_id)
    
    print("Original topic representations:")
    for topic_id in range(min(3, len(original_topics))):
        print(f"Topic {topic_id}: {original_topics[topic_id][:5]}")
    save_file(original_topics, f"{OUTPUT_DIR}/original_topics.json")
    
    print("\n=== Updating with Different N-gram Ranges ===")
    
    # Update with different n-gram ranges
    n_gram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams and bigrams"),
        ((1, 3), "Unigrams, bigrams, and trigrams"),
        ((2, 2), "Bigrams only")
    ]
    
    for n_gram_range, description in n_gram_configs:
        print(f"\n{description}:")
        model_updated = update_topic_representation(
            model, docs, topics,
            n_gram_range=n_gram_range,
            stop_words="english"
        )
        
        # Show top topics
        for topic_id in range(min(2, len(model_updated.get_topic_info()))):
            topic_words = model_updated.get_topic(topic_id)
            print(f"  Topic {topic_id}: {topic_words[:5]}")
            save_file({
                "n_gram_range": n_gram_range,
                "description": description,
                "count": len(topic_words),
                "topics": topic_words,
            }, f"{OUTPUT_DIR}/ngram/top_topics_{topic_id}_n_{n_gram_range}.json")
    
    print("\n=== Using Custom Vectorizer ===")
    
    # Create custom TF-IDF vectorizer
    custom_vectorizer = create_custom_vectorizer(
        vectorizer_type="tfidf",
        n_gram_range=(1, 2),
        min_df=1,
        max_df=0.9
    )
    
    model_tfidf = update_topic_representation(
        model, docs, topics,
        vectorizer=custom_vectorizer
    )
    
    print("Topics with TF-IDF vectorizer:")
    for topic_id in range(min(3, len(model_tfidf.get_topic_info()))):
        topic_words = model_tfidf.get_topic(topic_id)
        print(f"Topic {topic_id}: {topic_words[:5]}")
        save_file(topic_words, f"{OUTPUT_DIR}/tfidf/top_topics_{topic_id}.json")
        save_file({
            "count": len(topic_words),
            "topics": topic_words,
        }, f"{OUTPUT_DIR}/tfidf/top_topics_{topic_id}.json")
    
    print("\n=== Topic Keywords Analysis ===")
    for topic_id in range(min(3, len(model.get_topic_info()))):
        keywords = get_topic_keywords(model, topic_id, top_n=5)
        print(f"Topic {topic_id} keywords: {keywords}")
        save_file({
            "topic_id": topic_id,
            "count": len(keywords),
            "keywords": keywords
        }, f"{OUTPUT_DIR}/tfidf/topic_keywords_analysis_{topic_id}.json")
    
    print("\n=== Comparison of Representations ===")
    comparison = compare_topic_representations(
        model, docs, topics, original_topics
    )
    for key, value in comparison.items():
        if value:  # Check if not empty
            logger.log(
                key,
                ": ",
                format_json(value) if isinstance(value, (dict, list)) else value,
                colors=["DEBUG", "GRAY", "SUCCESS"]
            )
            save_file(
                {
                    "count": len(value) if hasattr(value, '__len__') else 1,
                    key: value,
                },
                f"{OUTPUT_DIR}/tfidf/comparison_{key}.json"
            )
    save_file(comparison, f"{OUTPUT_DIR}/tfidf/all_comparison_of_representations.json")
