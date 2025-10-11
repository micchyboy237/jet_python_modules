import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from jet.adapters.bertopic import BERTopic
from tqdm import tqdm

from jet.wordnet.text_chunker import chunk_texts
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

EMBED_MODEL = "embeddinggemma"

_sample_data_cache = None

def load_sample_data():
    """Load sample dataset from 20 newsgroups for topic modeling, with global cache."""
    global _sample_data_cache
    if _sample_data_cache is not None:
        return _sample_data_cache
    from sklearn.datasets import fetch_20newsgroups
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:100]
    documents = chunk_texts(
        documents,
        chunk_size=128,
        chunk_overlap=32,
        model=EMBED_MODEL,
    )
    _sample_data_cache = documents
    return documents

def example_get_topic():
    """Demonstrate retrieving topics for a BERTopic model."""
    logging.info("Starting get topic example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topics...")
    topics = set(topic_model.topics_)
    for topic in topics:
        topic_words = topic_model.get_topic(topic)
        logging.info(f"Topic {topic}: {', '.join([word[0] for word in topic_words[:5]])}")
    
    unknown_topic = topic_model.get_topic(500)
    logging.info(f"Unknown topic (500) result: {unknown_topic}")
    
    return topics

def example_get_topics():
    """Demonstrate retrieving all topic representations."""
    logging.info("Starting get topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving all topics...")
    topics = topic_model.get_topics()
    for topic_id, words in topics.items():
        logging.info(f"Topic {topic_id}: {', '.join([word[0] for word in words[:5]])}")
    
    return topics

def example_get_topic_freq():
    """Demonstrate retrieving topic frequencies."""
    logging.info("Starting get topic frequency example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topic frequencies...")
    topic_freq = topic_model.get_topic_freq()
    logging.info(f"Topic frequency DataFrame:\n{topic_freq.head()}")
    
    for topic in set(topic_model.topics_):
        freq = topic_model.get_topic_freq(topic)
        logging.info(f"Frequency for topic {topic}: {freq}")
    
    return topic_freq

def example_get_representative_docs():
    """Demonstrate retrieving representative documents for topics."""
    logging.info("Starting get representative documents example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving representative documents...")
    all_docs = topic_model.get_representative_docs()
    for topic_id, docs in all_docs.items():
        logging.info(f"Topic {topic_id} representative docs: {docs}")
    
    return all_docs

def example_get_topic_info():
    """Demonstrate retrieving topic information."""
    logging.info("Starting get topic info example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topic information...")
    topic_info = topic_model.get_topic_info()
    logging.info(f"Topic info DataFrame:\n{topic_info.head()}")
    
    for topic in set(topic_model.topics_):
        info = topic_model.get_topic_info(topic)
        logging.info(f"Info for topic {topic}:\n{info}")
    
    return topic_info

def example_generate_topic_labels():
    """Demonstrate generating topic labels."""
    logging.info("Starting generate topic labels example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Generating topic labels...")
    labels = topic_model.generate_topic_labels(topic_prefix=False)
    for label in labels:
        logging.info(f"Generated label: {label}")
    
    labels_with_prefix = topic_model.generate_topic_labels(nr_words=1, topic_prefix=False)
    logging.info(f"Single-word labels: {labels_with_prefix}")
    
    return labels

def example_set_labels():
    """Demonstrate setting custom topic labels."""
    logging.info("Starting set topic labels example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Setting custom topic labels...")
    labels = topic_model.generate_topic_labels()
    topic_model.set_topic_labels(labels)
    logging.info(f"Set custom labels: {topic_model.custom_labels_[:5]}")
    
    custom_labels = {1: "Custom Topic 1", 2: "Custom Topic 2"}
    topic_model.set_topic_labels(custom_labels)
    logging.info(f"Updated custom labels: {topic_model.custom_labels_}")
    
    return topic_model.custom_labels_

def example_update_topics():
    """Demonstrate updating topic representations."""
    logging.info("Starting update topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Updating topics with n-grams...")
    old_ctfidf = topic_model.c_tf_idf_
    old_topics = topic_model.topics_
    with tqdm(total=len(documents), desc="Updating topics") as pbar:
        topic_model.update_topics(documents, n_gram_range=(1, 3))
        pbar.update(len(documents))
    
    logging.info(f"Old c-tf-idf shape: {old_ctfidf.shape}, New c-tf-idf shape: {topic_model.c_tf_idf_.shape}")
    
    return topic_model

def example_extract_topics():
    """Demonstrate extracting topics with custom topic assignments."""
    logging.info("Starting extract topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    nr_topics = 5
    topics = np.random.randint(-1, nr_topics - 1, len(documents))
    docs_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    
    logging.info("Extracting topics...")
    with tqdm(total=len(documents), desc="Extracting topics") as pbar:
        topic_model._update_topic_size(docs_df)
        topic_model._extract_topics(docs_df)
        pbar.update(len(documents))
    
    freq = topic_model.get_topic_freq()
    logging.info(f"Topic frequency DataFrame:\n{freq}")
    
    return freq

def example_extract_topics_custom_cv():
    """Demonstrate extracting topics with custom vectorizer."""
    logging.info("Starting extract topics with custom vectorizer example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    nr_topics = 5
    topics = np.random.randint(-1, nr_topics - 1, len(documents))
    docs_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    
    logging.info("Setting custom vectorizer...")
    cv = CountVectorizer(ngram_range=(1, 2))
    topic_model.vectorizer_model = cv
    
    logging.info("Extracting topics with custom vectorizer...")
    with tqdm(total=len(documents), desc="Extracting topics") as pbar:
        topic_model._update_topic_size(docs_df)
        topic_model._extract_topics(docs_df)
        pbar.update(len(documents))
    
    freq = topic_model.get_topic_freq()
    logging.info(f"Topic frequency DataFrame:\n{freq}")
    
    return freq

def example_topic_reduction():
    """Demonstrate reducing the number of topics."""
    logging.info("Starting topic reduction example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    reduced_topics = 3
    logging.info(f"Reducing to {reduced_topics} topics...")
    with tqdm(total=len(documents), desc="Reducing topics") as pbar:
        topic_model.reduce_topics(documents, nr_topics=reduced_topics)
        pbar.update(len(documents))
    
    freq = topic_model.get_topic_freq()
    logging.info(f"Reduced topic frequency DataFrame:\n{freq}")
    
    return freq

def example_find_topics():
    """Demonstrate finding topics similar to a query."""
    logging.info("Starting find topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    query = "car"
    logging.info(f"Searching for topics similar to: {query}")
    similar_topics, similarity = topic_model.find_topics(query)
    for topic, sim in zip(similar_topics, similarity):
        words = topic_model.get_topic(topic)[:5]
        logging.info(f"Topic {topic} (Similarity: {sim:.2f}): {', '.join([word[0] for word in words])}")
    
    return similar_topics, similarity

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = example_get_topic()
    save_file(results, f"{OUTPUT_DIR}/01_get_topic_results.json")
    results = example_get_topics()
    save_file(results, f"{OUTPUT_DIR}/02_get_topics_results.json")
    results = example_get_topic_freq()
    save_file(results, f"{OUTPUT_DIR}/03_get_topic_freq_results.json")
    results = example_get_representative_docs()
    save_file(results, f"{OUTPUT_DIR}/04_get_representative_docs_results.json")
    results = example_get_topic_info()
    save_file(results, f"{OUTPUT_DIR}/05_get_topic_info_results.json")
    results = example_generate_topic_labels()
    save_file(results, f"{OUTPUT_DIR}/06_generate_topic_labels_results.json")
    results = example_set_labels()
    save_file(results, f"{OUTPUT_DIR}/07_set_labels_results.json")
    results = example_update_topics()
    save_file(results, f"{OUTPUT_DIR}/08_update_topics_results.json")
    results = example_extract_topics()
    save_file(results, f"{OUTPUT_DIR}/09_extract_topics_results.json")
    results = example_extract_topics_custom_cv()
    save_file(results, f"{OUTPUT_DIR}/10_extract_topics_custom_cv_results.json")
    results = example_topic_reduction()
    save_file(results, f"{OUTPUT_DIR}/11_topic_reduction_results.json")
    results = example_find_topics()
    save_file(results, f"{OUTPUT_DIR}/12_find_topics_results.json")
    logging.info("All representation usage examples completed successfully.")
