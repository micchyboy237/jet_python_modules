import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

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
    
    custom_labels = {1: "Changed Topic 1", 3: "New Topic"}
    topic_model.set_topic_labels(custom_labels)
    logging.info(f"Further updated custom labels: {topic_model.custom_labels_}")
    
    return topic_model.custom_labels_

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_generate_topic_labels()
    example_set_labels()
    logging.info("Labels usage examples completed successfully.")
