import numpy as np
from typing import Tuple, List, Optional
from jet.adapters.bertopic import BERTopic
from jet.libs.bertopic.jet_examples.mock import load_sample_data

def reduce_topic_count(
    topic_model: BERTopic,
    docs: List[str],
    nr_topics: int
) -> Tuple[BERTopic, List[int], Optional[np.ndarray]]:
    """
    Reduce the number of topics in an existing model.
    Args:
        topic_model: The fitted BERTopic model.
        docs: List of documents used for training.
        nr_topics: Target number of topics to reduce to.
    Returns:
        tuple: Updated topic_model, new_topics, new_probs
    """
    # Call reduce_topics with only supported parameters
    topic_model.reduce_topics(
        docs=docs,
        nr_topics=nr_topics
    )
    # Retrieve updated topics and probabilities from the model
    new_topics = topic_model.topics_
    new_probs = topic_model.probabilities_
    return topic_model, new_topics, new_probs


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents
    # docs = [
    #     "I love machine learning and data science.",
    #     "COVID-19 has impacted global economies.",
    #     "Quantum computing might revolutionize encryption.",
    #     "The stock market had a wild ride this week.",
    #     "Machine learning algorithms are becoming more sophisticated.",
    #     "The pandemic changed how we work and live.",
    #     "Cryptocurrency markets are highly volatile.",
    #     "Artificial intelligence is transforming healthcare."
    # ]
    docs = load_sample_data()
    
    # First fit a model with many topics
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    print("Original topics:")
    print(model.get_topic_info())
    
    # Reduce to fewer topics
    target_topics = 5
    model_reduced, new_topics, new_probs = reduce_topic_count(
        model, docs, nr_topics=target_topics
    )
    print("\nReduced topics:")
    print(model_reduced.get_topic_info())
    
    # Show topic assignments and new probabilities
    print("\nTopic assignments and probabilities:")
    for i, (doc, topic, prob) in enumerate(
        zip(docs, new_topics, new_probs if new_probs is not None else [None] * len(docs))
    ):
        print(f"Doc {i} | Prob: {prob}\nTopic: {topic}\n    {doc}\n")
