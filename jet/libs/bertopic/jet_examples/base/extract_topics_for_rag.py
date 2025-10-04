from typing import List, Dict, Any, Tuple
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP  # Import UMAP explicitly

def extract_topics_for_rag(
    docs: List[str],
    query: str,
    top_k_topics: int = 3,
    nr_topics: str = "auto",
    min_prob: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
    """
    Extract topics and doc-topic mappings from documents using BERTopic for RAG augmentation.
    
    Args:
        docs: List of unstructured/structured docs (flattened text).
        query: User query for guided topic seeding.
        top_k_topics: Number of top topics to return.
        nr_topics: BERTopic nr_topics param.
        min_prob: Minimum probability for doc-topic assignment.
    
    Returns:
        Tuple of topic dicts, topic summaries, and doc-topic mappings.
    """
    if len(docs) < 2:
        raise ValueError("At least 2 documents are required for topic modeling.")

    # Embeddings model (reusable)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Custom UMAP model for small datasets
    n_components = min(2, len(docs) - 1)  # Ensure n_components < N
    umap_model = UMAP(
        n_neighbors=min(2, len(docs) - 1),  # Adjust for small datasets
        n_components=n_components,
        random_state=42,  # Reproducibility
        metric="cosine",  # Suitable for text embeddings
        low_memory=True  # Reduce memory usage for M1
    )
    
    # Initialize and fit BERTopic with custom UMAP
    topic_model = BERTopic(
        embedding_model=embedding_model,
        nr_topics=nr_topics,
        verbose=True,
        umap_model=umap_model
    )
    topics, probs = topic_model.fit_transform(docs)
    
    # Seed topics with query for alignment (new in v0.16)
    seed_topics = [[query]]
    topic_model.update_topics(docs, topics=seed_topics)
    
    # Recompute topics and probs after update_topics to reflect changes
    topics, probs = topic_model.transform(docs)
    
    # Get top topics with representations
    topic_info = topic_model.get_topic_info()
    top_topics = topic_info.head(top_k_topics)
    
    # Generate summaries (using built-in c-TF-IDF)
    summaries = [
        topic_model.get_topic(i)["Name"] + ": " + ", ".join([word[0] for word in topic_model.get_topic(i)[:5]])
        for i in top_topics["Topic"]
        if i != -1  # Exclude outliers
    ]
    
    topic_dicts = [
        {
            "topic_id": row["Topic"],
            "count": row["Count"],
            "name": topic_model.get_topic(row["Topic"])["Name"] if hasattr(topic_model.get_topic(row["Topic"]), "Name") else "Unnamed"
        }
        for _, row in top_topics.iterrows()
        if row["Topic"] != -1
    ]
    
    # New: Map documents to topics with confidence
    doc_topic_mappings = [
        {
            "doc_id": i,
            "doc_text": doc[:100] + "...",
            "topic_id": topics[i],
            "probability": float(probs[i]) if probs is not None and i < len(probs) else 0.0
        }
        for i, doc in enumerate(docs)
        if topics[i] != -1 and (probs is None or probs[i] >= min_prob)
    ]
    
    return topic_dicts, summaries, doc_topic_mappings

# Example usage
if __name__ == "__main__":
    docs: List[str] = [
        "Climate change is accelerating due to fossil fuels and deforestation.",
        "Renewable energy sources like solar and wind are essential for sustainability.",
        "Global warming impacts include rising sea levels and extreme weather.",
        "Carbon emissions from industries must be regulated internationally.",
        "Biodiversity loss is a direct consequence of habitat destruction."
    ]
    query: str = "environmental impacts"

    topics, summaries, doc_mappings = extract_topics_for_rag(docs, query)
    print("Topics:", topics)
    print("Summaries:", summaries)
    print("Doc Mappings:", doc_mappings)
