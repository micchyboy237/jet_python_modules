from typing import List, Tuple
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import pandas as pd
import os
import shutil

from jet.logger import logger

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

# Sample documents for demonstration
SAMPLE_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing",
    "BERTopic is a powerful topic modeling tool",
    "Natural language models are improving rapidly",
    "The dog sleeps while the fox runs"
]

def example_init_bertopic() -> BERTopic:
    """
    Demonstrates initialization of BERTopic with custom parameters.
    - Configures a multilingual model with specific n-gram range and topic size.
    - Uses a custom CountVectorizer for preprocessing.
    """
    vectorizer = CountVectorizer(stop_words="english")
    model = BERTopic(
        language="multilingual",
        top_n_words=5,
        n_gram_range=(1, 2),
        min_topic_size=2,
        vectorizer_model=vectorizer,
        verbose=True
    )
    return model

def example_fit_transform() -> Tuple[List[int], np.ndarray]:
    """
    Demonstrates fitting the BERTopic model and transforming documents to topics.
    - Fits the model on sample documents and retrieves topic assignments and probabilities.
    - Shows how to handle document-to-topic mapping.
    """
    model = example_init_bertopic()
    topics, probabilities = model.fit_transform(documents=SAMPLE_DOCS)
    # Example output inspection
    print(f"Topics assigned: {topics}")
    print(f"Probabilities shape: {probabilities.shape if probabilities is not None else None}")
    return topics, probabilities

def example_get_topic_info() -> pd.DataFrame:
    """
    Demonstrates retrieving topic information after fitting the model.
    - Shows topic sizes and representative words for each topic.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    topic_info = model.get_topic_info()
    print(f"Topic info:\n{topic_info}")
    return topic_info

def example_get_representative_docs(topic: int = 0) -> List[str]:
    """
    Demonstrates retrieving representative documents for a specific topic.
    - Useful for understanding what documents best represent a topic.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    docs = model.get_representative_docs(topic=topic)
    print(f"Representative docs for topic {topic}: {docs}")
    return docs

def example_visualize_topics() -> None:
    """
    Demonstrates visualizing the intertopic distance map.
    - Requires fitting the model first to generate topic embeddings.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    fig = model.visualize_topics(title="Sample Topic Distance Map")
    fig.show()

def example_update_topics() -> None:
    """
    Demonstrates updating topic representations with new parameters.
    - Shows how to refine topics after initial fitting.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    new_vectorizer = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    model.update_topics(docs=SAMPLE_DOCS, vectorizer_model=new_vectorizer, top_n_words=3)
    topic_info = model.get_topic_info()
    print(f"Updated topic info:\n{topic_info}")

def example_reduce_topics() -> None:
    """
    Demonstrates reducing the number of topics after fitting.
    - Shows how to consolidate topics to a specified number.
    """
    model = example_init_bertopic()
    model.fit(documents=SAMPLE_DOCS)
    model.reduce_topics(docs=SAMPLE_DOCS, nr_topics=2)
    topic_info = model.get_topic_info()
    print(f"Reduced topic info:\n{topic_info}")