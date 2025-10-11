from jet.adapters.bertopic import BERTopic


def find_similar_topics(
    topic_model: BERTopic,
    term: str,
    top_n: int = 10
) -> tuple[list[int], list[float]]:
    """
    Given a keyword or term, find the top_n most semantically similar topics and their similarity scores.
    
    Args:
        topic_model: The fitted BERTopic model
        term: The search term to find similar topics for
        top_n: Number of similar topics to return
        
    Returns:
        tuple: List of topic IDs and their similarity scores
    """
    topics, similarity = topic_model.find_topics(term, top_n=top_n)
    return topics, similarity


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents covering different topics
    docs = [
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Data science involves statistics, programming, and domain expertise.",
        "COVID-19 pandemic has changed global health and economy.",
        "Vaccines and medical research are crucial for public health.",
        "Quantum computing could break current encryption methods.",
        "Cryptocurrency and blockchain technology are emerging trends.",
        "Climate change is affecting weather patterns worldwide.",
        "Renewable energy sources like solar and wind are growing.",
        "Stock market volatility affects investor confidence.",
        "Economic policies influence inflation and employment rates."
    ]
    
    # Fit the model
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    print("Topic information:")
    print(model.get_topic_info())
    
    # Find topics similar to different terms
    search_terms = ["data", "health", "technology", "economy"]
    
    for term in search_terms:
        print(f"\nSearching for topics similar to '{term}':")
        similar_topics, sim_scores = find_similar_topics(model, term, top_n=3)
        
        for topic_id, score in zip(similar_topics, sim_scores):
            topic_words = model.get_topic(topic_id)
            print(f"  Topic {topic_id} (similarity: {score:.3f}): {topic_words[:5]}")
