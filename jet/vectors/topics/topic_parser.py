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
    embedding_model: str = "all-MiniLM-L12-v2",
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


if __name__ == "__main__":
    # Sample documents
    print("Using sample documents...")
    docs: List[str] = [
        "The stock market crashed today as tech stocks took a hit.",
        "A new study shows the health benefits of a Mediterranean diet.",
        "NASA plans to launch a new satellite to monitor climate change.",
        "Python is a popular programming language for data science.",
        "The local team won the championship after a thrilling final."
    ]

    # Fit BERTopic model
    print("Fitting BERTopic model...")
    try:
        topic_model: BERTopic = configure_topic_model()
        topics: List[int]
        probs: List[np.ndarray]
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        print(f"Error fitting model: {e}")
        exit(1)

    # Get topic info as a DataFrame
    print("Getting topic info...")
    topic_info: pd.DataFrame = topic_model.get_topic_info()

    # Display the first few topics and their details
    print("Top Topics:")
    print(topic_info[['Topic', 'Name', 'Count']].head(
        10).to_string(index=False))

    # Show a sample of documents with their assigned topics and probabilities
    sample_df: pd.DataFrame = create_topic_df(docs, topics, probs)
    print("\nSample Documents, Their Assigned Topics, and Probabilities:")
    print(sample_df.to_string(index=False, max_colwidth=60))
