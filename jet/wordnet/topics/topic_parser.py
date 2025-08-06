from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def configure_topic_model(
    n_components: int = 2,
    n_neighbors: int = 3,
    min_dist: float = 0.1,
    min_cluster_size: int = 2,
    min_samples: int = 1,
    embedding_model: str = "all-MiniLM-L6-v2",
    random_state: Optional[int] = 42
) -> BERTopic:
    """Configure a BERTopic model with UMAP and HDBSCAN."""
    umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                      min_dist=min_dist, random_state=random_state)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples)
    vectorizer_model = CountVectorizer(
        stop_words="english")  # Remove stopwords
    return BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        embedding_model=embedding_model,
        min_topic_size=min_cluster_size,
        vectorizer_model=vectorizer_model
    )


def create_topic_df(docs: List[str], topics: List[int], probs: List[np.ndarray]) -> pd.DataFrame:
    """Create a DataFrame with documents, topics, and probabilities."""
    return pd.DataFrame({
        "Document": docs,
        "Assigned Topic": topics,
        "Topic Probability": [round(prob.max() if prob.size > 0 else 0.0, 4) for prob in probs]
    })
