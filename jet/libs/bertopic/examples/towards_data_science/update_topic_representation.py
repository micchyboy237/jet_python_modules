import os
import shutil
from typing import TypedDict

from jet.adapters.bertopic import BERTopic
from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.logger import logger
from jet.transformers.formatters import format_json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Define TypedDict for topic/keyword representation
class TopicRepresentation(TypedDict):
    word: str
    score: float


# Define TypedDict for topic comparison entries
class TopicComparisonEntry(TypedDict):
    topic_id: int
    words: list[TopicRepresentation]


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "generated",
    os.path.splitext(os.path.basename(__file__))[0],
)

CHUNK_SIZE = 150
CHUNK_OVERLAP = 40

OUTPUT_DIR = f"{OUTPUT_DIR}/chunked_{CHUNK_SIZE}_{CHUNK_OVERLAP}"
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def update_topic_representation(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    n_gram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    vectorizer: CountVectorizer | None = None,
    min_df: int = 2,
    max_df: float = 0.8,
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
            max_df=max_df,
        )

    topic_model.update_topics(docs=docs, topics=topics, vectorizer_model=vectorizer)
    return topic_model


def create_custom_vectorizer(
    vectorizer_type: str = "count",
    n_gram_range: tuple[int, int] = (1, 2),
    stop_words: str | None = "english",
    min_df: int = 2,
    max_df: float = 0.8,
    max_features: int | None = None,
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
        "max_features": max_features,
    }

    if vectorizer_type.lower() == "tfidf":
        return TfidfVectorizer(**common_params)
    else:
        return CountVectorizer(**common_params)


def compare_topic_representations(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    original_topics: dict = None,
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

    comparison = {"current_topics": current_topics, "n_topics": len(current_topics)}

    if original_topics:
        comparison["original_topics"] = original_topics
        comparison["changed_topics"] = []

        for topic_id in current_topics:
            if topic_id in original_topics:
                if current_topics[topic_id] != original_topics[topic_id]:
                    comparison["changed_topics"].append(topic_id)

    return comparison


def get_topic_keywords(
    topic_model: BERTopic, topic_id: int, top_n: int | None = None
) -> list[tuple[str, float]]:
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
    return topic_words[:top_n] if topic_words else []


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform

    model = "nomic-embed-text"
    docs = load_sample_data(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    save_file(docs, f"{OUTPUT_DIR}/documents.json")

    print("=== Initial Topic Model ===")
    # Ensure topics are generated for the same docs
    model, topics, probs = topic_model_fit_transform(
        docs,  # Use the same docs from load_sample_data
        calculate_probabilities=True,
        embedding_model=model,
    )

    # Verify lengths match
    if len(docs) != len(topics):
        raise ValueError(f"Mismatch in lengths: docs={len(docs)}, topics={len(topics)}")

    # Store original topics for comparison
    original_topics = {}
    for topic_id in range(len(model.get_topic_info())):
        original_topics[topic_id] = model.get_topic(topic_id)

    print("Original topic representations:")
    for topic_id in range(min(3, len(original_topics))):
        print(f"Topic {topic_id}: {original_topics[topic_id][:5]}")
    # Convert original topics to list of typed dicts
    original_topics_list = [
        TopicComparisonEntry(
            topic_id=topic_id,
            words=[
                TopicRepresentation(word=word, score=score)
                for word, score in (words if words else [])
            ],
        )
        for topic_id, words in original_topics.items()
    ]
    save_file(original_topics_list, f"{OUTPUT_DIR}/original_topics.json")

    print("\n=== Updating with Different N-gram Ranges ===")

    # Update with different n-gram ranges
    n_gram_configs = [
        ((1, 1), "Unigrams only"),
        ((1, 2), "Unigrams and bigrams"),
        ((1, 3), "Unigrams, bigrams, and trigrams"),
        ((2, 2), "Bigrams only"),
    ]

    for n_gram_range, description in n_gram_configs:
        print(f"\n{description}:")
        model_updated = update_topic_representation(
            model, docs, topics, n_gram_range=n_gram_range, stop_words="english"
        )

        # Show top topics
        for topic_id in model_updated.get_topic_info():
            topics_words = model_updated.get_topic(topic_id)
            if not topics_words:
                topics_words = []
            print(f" Topic {topic_id}: {topics_words[:5]}")
            # Convert topics to list of typed dicts
            topics_words_list = [
                TopicRepresentation(word=word, score=score)
                for word, score in topics_words
            ]
            save_file(
                {
                    "n_gram_range": n_gram_range,
                    "description": description,
                    "count": len(topics_words_list),
                    "topics": topics_words_list,
                },
                f"{OUTPUT_DIR}/ngram/top_topics_{topic_id}/n_{n_gram_range[0]}_{n_gram_range[1]}.json",
            )

    print("\n=== Using TF-IDF Vectorizer ===")

    # Create custom TF-IDF vectorizer
    custom_vectorizer = create_custom_vectorizer(
        vectorizer_type="tfidf", n_gram_range=(1, 2), min_df=1, max_df=0.9
    )

    model_tfidf = update_topic_representation(
        model, docs, topics, vectorizer=custom_vectorizer
    )

    print("Topics with TF-IDF vectorizer:")
    for topic_id in range(len(model_tfidf.get_topic_info())):
        topics_words = model_tfidf.get_topic(topic_id)
        if not topics_words:
            topics_words = []
        print(f"Topic {topic_id}: {topics_words[:5]}")
        # Convert topics to list of typed dicts
        topics_words_list = [
            TopicRepresentation(word=word, score=score) for word, score in topics_words
        ]
        save_file(
            {
                "topic_id": topic_id,
                "count": len(topics_words_list),
                "topics": topics_words_list,
            },
            f"{OUTPUT_DIR}/tfidf/top_topics/{topic_id}.json",
        )

    print("\n=== Topic Representations ===")
    for topic_id in range(len(model_tfidf.get_topic_info())):
        representations = model_tfidf.get_topic(topic_id, full=True)
        if not representations or "Main" not in representations:
            representations_list = []
        else:
            # Convert tuples to typed dictionaries
            representations_list = [
                TopicRepresentation(word=word, score=score)
                for word, score in representations["Main"]
            ]
        print(f"Topic Representation {topic_id}: {representations_list[:5]}")
        save_file(
            representations_list,
            f"{OUTPUT_DIR}/tfidf/top_representations/{topic_id}.json",
        )

    print("\n=== Comparison of Representations ===")
    comparison = compare_topic_representations(
        model_tfidf, docs, topics, original_topics
    )
    for key, value in comparison.items():
        if value:  # Check if not empty
            logger.log(
                key,
                ": ",
                format_json(value) if isinstance(value, (dict, list)) else value,
                colors=["DEBUG", "GRAY", "SUCCESS"],
            )
            # Convert comparison topics to list of typed dicts
            if key in ["current_topics", "original_topics"]:
                value_list = [
                    TopicComparisonEntry(
                        topic_id=topic_id,
                        words=[
                            TopicRepresentation(word=word, score=score)
                            for word, score in (words if words else [])
                        ],
                    )
                    for topic_id, words in value.items()
                ]
            else:
                value_list = (
                    value  # Keep n_topics (int) and changed_topics (list[int]) as is
                )
            save_file(
                {
                    "count": len(value_list) if hasattr(value_list, "__len__") else 1,
                    key: value_list,
                },
                f"{OUTPUT_DIR}/tfidf/comparisons/{key}.json",
            )
