from jet.adapters.bertopic import BERTopic


def reduce_topic_count(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    probs,
    nr_topics: int
):
    """
    Reduce the number of topics in an existing model.
    
    Args:
        topic_model: The fitted BERTopic model
        docs: List of documents used for training
        topics: List of topic assignments for each document
        probs: Topic probabilities for each document
        nr_topics: Target number of topics to reduce to
        
    Returns:
        tuple: Updated topic_model, new_topics, new_probs
    """
    new_topics, new_probs = topic_model.reduce_topics(
        docs, topics, probs, nr_topics=nr_topics
    )
    return topic_model, new_topics, new_probs


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents
    docs = [
        "I love machine learning and data science.",
        "COVID-19 has impacted global economies.",
        "Quantum computing might revolutionize encryption.",
        "The stock market had a wild ride this week.",
        "Machine learning algorithms are becoming more sophisticated.",
        "The pandemic changed how we work and live.",
        "Cryptocurrency markets are highly volatile.",
        "Artificial intelligence is transforming healthcare."
    ]
    
    # First fit a model with many topics
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    print("Original topics:")
    print(model.get_topic_info())
    
    # Reduce to fewer topics
    model, new_topics, new_probs = reduce_topic_count(model, docs, topics, probs, nr_topics=3)
    print("\nReduced topics:")
    print(model.get_topic_info())
    
    # Show topic assignments
    print("\nTopic assignments:")
    for i, (doc, topic) in enumerate(zip(docs, new_topics)):
        print(f"Doc {i}: Topic {topic} - {doc}")
