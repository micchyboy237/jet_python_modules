import numpy as np
from typing import Optional, Tuple, List
from jet.adapters.bertopic import BERTopic
from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding


def topic_model_fit_transform(
    docs: list[str],
    embedding_model: str = "embeddinggemma",
    language: str = "english",
    nr_topics: int | str = "auto",
    calculate_probabilities: bool = False,
    precomputed_embeddings: Optional[np.ndarray] = None,
) -> Tuple[BERTopic, List[int], Optional[np.ndarray]]:
    """
    Fit a BERTopic model to the documents and return topics + probabilities.
    
    Args:
        docs: List of documents to analyze
        embedding_model: Name of the sentence transformer model to use
        language: Language for stop words and preprocessing
        nr_topics: Number of topics to extract ("auto" for automatic)
        calculate_probabilities: Whether to calculate topic probabilities
        precomputed_embeddings: Pre-computed embeddings (optional)
        
    Returns:
        tuple: (topic_model, topics, probabilities)
    """
    # Create BERTopic model with specified parameters
    topic_model = BERTopic(
        embedding_model=embedding_model,
        language=language,
        nr_topics=nr_topics,
        calculate_probabilities=calculate_probabilities,
    )
    
    # Fit and transform the model
    if precomputed_embeddings is not None:
        topics, probs = topic_model.fit_transform(docs, embeddings=precomputed_embeddings)
    else:
        topics, probs = topic_model.fit_transform(docs)
    
    return topic_model, topics, probs


def precompute_embeddings(
    docs: list[str],
    embedding_model: str = "embeddinggemma"
) -> np.ndarray:
    """
    Pre-compute embeddings for documents to speed up multiple model runs.
    
    Args:
        docs: List of documents
        embedding_model: Name of the sentence transformer model
        
    Returns:
        numpy array of embeddings
    """
    embedder = LlamacppEmbedding(model=embedding_model)
    embeddings = embedder(docs)
    return embeddings


def get_topic_statistics(topic_model: BERTopic, topics: List[int]) -> dict:
    """
    Get comprehensive statistics about the topics.
    
    Args:
        topic_model: Fitted BERTopic model
        topics: List of topic assignments
        
    Returns:
        dict: Topic statistics
    """
    topic_info = topic_model.get_topic_info()
    
    stats = {
        "n_topics": len(topic_info),
        "n_documents": len(topics),
        "topic_distribution": dict(zip(*np.unique(topics, return_counts=True))),
        "outlier_percentage": (topics.count(-1) / len(topics)) * 100,
        "avg_docs_per_topic": len(topics) / len(topic_info),
        "topic_info": topic_info
    }
    
    return stats


if __name__ == "__main__":
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
        "Economic policies influence inflation and employment rates.",
        "Deep learning neural networks require large datasets.",
        "Natural language processing is advancing rapidly.",
        "Computer vision applications are expanding in healthcare.",
        "Robotics and automation are transforming manufacturing.",
        "Internet of Things devices are becoming more prevalent."
    ]
    
    print("=== Basic BERTopic Model Training ===")
    model, topics, probs = topic_model_fit_transform(
        docs, 
        calculate_probabilities=True,
    )
    
    print("Topic Information:")
    print(model.get_topic_info())
    
    print("\nTopic Statistics:")
    stats = get_topic_statistics(model, topics)
    for key, value in stats.items():
        if key != "topic_info":  # Skip the large topic_info dict
            print(f"  {key}: {value}")
    
    print("\nDocument-Topic Assignments:")
    for i, (doc, topic) in enumerate(zip(docs, topics)):
        print(f"Doc {i}: Topic {topic} - {doc[:50]}...")
    
    print("\n=== Using Pre-computed Embeddings ===")
    # Pre-compute embeddings for efficiency
    embeddings = precompute_embeddings(docs)
    print(f"Pre-computed embeddings shape: {embeddings.shape}")
    
    # Use pre-computed embeddings
    model2, topics2, probs2 = topic_model_fit_transform(
        docs,
        precomputed_embeddings=embeddings,
        calculate_probabilities=True
    )
    
    print("Model with pre-computed embeddings:")
    print(model2.get_topic_info())
    
    print("\n=== Topic Details ===")
    for topic_id in range(min(3, len(model.get_topic_info()))):
        topic_words = model.get_topic(topic_id)
        print(f"Topic {topic_id}: {topic_words[:5]}")
