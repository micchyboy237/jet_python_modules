from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from typing import List, Optional, TypedDict


def configure_topic_model(
    n_components: int = 2,
    n_neighbors: int = 3,
    min_dist: float = 0.1,
    min_cluster_size: int = 2,
    min_samples: int = 2,  # Increased from 1
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: Optional[int] = 42
) -> BERTopic:
    """Configure a BERTopic model with UMAP and HDBSCAN."""
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                      min_dist=min_dist, random_state=random_state)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True  # Enable prediction data for probabilities
    )
    vectorizer_model = CountVectorizer(stop_words="english")
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        min_topic_size=min_cluster_size,
        vectorizer_model=vectorizer_model,
        calculate_probabilities=True  # Explicitly enable probabilities
    )


def create_topic_df(docs: List[str], topics: List[int], probs: List[np.ndarray]) -> pd.DataFrame:
    """Create a DataFrame with documents, topics, and probabilities."""
    return pd.DataFrame({
        "Document": docs,
        "Assigned Topic": topics,
        "Topic Probability": [round(prob.max() if prob.size > 0 else 0.0, 4) for prob in probs]
    })


class TopicInfo(TypedDict):
    topic_id: int
    topic_name: str
    document_count: int
    top_words: List[str]
    representative_docs: List[str]
    doc_probabilities: List[float]

def extract_topics(
    topic_model: BERTopic,
    docs: List[str],
    topics: List[int],
    probs: List[np.ndarray],
    num_top_words: int = 5,
    num_representative_docs: int = 3
) -> List[TopicInfo]:
    """Extract topic information with structured output, including probabilities.

    Args:
        topic_model: Fitted BERTopic model.
        docs: List of input documents.
        topics: List of topic IDs for each document.
        probs: List of probability arrays for each document.
        num_top_words: Number of top words to extract per topic.
        num_representative_docs: Number of representative documents to extract.

    Returns:
        List of TopicInfo dictionaries containing topic details.
    """
    topic_info: pd.DataFrame = topic_model.get_topic_info()
    result: List[TopicInfo] = []
    
    for _, row in topic_info.iterrows():
        topic_id = row['Topic']
        if topic_id == -1:  # Skip outlier topic
            continue
            
        # Get top words, handle potential empty topic
        topic_words = topic_model.get_topic(topic_id) or []
        top_words = [word for word, _ in topic_words[:num_top_words]] if topic_words else []
        
        # Get representative documents based on highest probabilities
        doc_indices = [
            i for i in range(len(docs)) 
            if topics[i] == topic_id
        ]
        if doc_indices:
            # Sort documents by probability (if probs available)
            doc_probs = []
            for i in doc_indices:
                if probs[i] is None or not isinstance(probs[i], np.ndarray):
                    # Handle scalar or None probabilities
                    prob = 1.0 if topics[i] == topic_id else 0.0
                else:
                    prob = probs[i][topic_id] if topic_id < len(probs[i]) else 0.0
                doc_probs.append((i, prob))
            doc_probs.sort(key=lambda x: x[1], reverse=True)
            top_doc_indices = [i for i, _ in doc_probs[:num_representative_docs]]
            rep_docs = [docs[i] for i in top_doc_indices]
            rep_probs = [round(float(prob), 4) for _, prob in doc_probs[:num_representative_docs]]
        else:
            rep_docs = []
            rep_probs = []
        
        # Create cleaner topic name by joining top words
        topic_name = row['Name'] if top_words else "Unnamed Topic"
        if top_words:
            topic_name = "_".join(top_words[:3]).replace(" ", "_")
        
        result.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'document_count': row['Count'],
            'top_words': top_words,
            'representative_docs': rep_docs,
            'doc_probabilities': rep_probs
        })
    
    return result
