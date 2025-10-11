import numpy as np
from typing import Tuple, List, Optional
from jet.adapters.bertopic import BERTopic
from jet.libs.bertopic.examples.mock import load_sample_data
from jet.file.utils import save_file
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

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
    # Create a list of typed dict for original_topics using topics and probs info
    original_topics = [
        {
            "doc_index": i,
            "topic_id": topic,
            "probability": probs[i] if probs is not None else None,
            "text": docs[i]
        }
        for i, topic in enumerate(topics)
    ]
    save_file(original_topics, f"{OUTPUT_DIR}/original_topics.json")
    
    # Reduce to fewer topics
    target_topics = 5
    model_reduced, new_topics, new_probs = reduce_topic_count(
        model, docs, nr_topics=target_topics
    )
    # Show topic assignments and new probabilities
    print("\nTopic assignments and probabilities (showing up to 5):")
    for i, (doc, topic, prob) in enumerate(
        zip(docs, new_topics, new_probs if new_probs is not None else [None] * len(docs))
    ):
        if i >= 5:
            break
        print(f"Doc {i} | Prob: {prob}\nTopic: {topic}\n    {doc}\n")

    print("\nReduced topics:")
    reduced_topics = [
        {
            "doc_index": i,
            "topic_id": topic,
            "probability": new_probs[i] if new_probs is not None else None,
            "text": docs[i]
        }
        for i, topic in enumerate(new_topics)
    ]
    print(model_reduced.get_topic_info())
    save_file(reduced_topics, f"{OUTPUT_DIR}/reduced_topics.json")
    
