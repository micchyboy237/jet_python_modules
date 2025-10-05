import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_merge_topics():
    """Demonstrate merging topics in a BERTopic model."""
    logging.info("Starting merge topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Merging topics [1, 2]...")
    nr_topics_before = len(set(topic_model.topics_))
    topics_to_merge = [1, 2]
    topic_model.merge_topics(documents, topics_to_merge)
    logging.info(f"Number of topics reduced from {nr_topics_before} to {len(set(topic_model.topics_))}")    
    
    logging.info("Merging topics [1, 2] again...")
    topic_model.merge_topics(documents, topics_to_merge)
    logging.info(f"Number of topics further reduced to {len(set(topic_model.topics_))}")    
    
    return topic_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_merge_topics()
    logging.info("Merge usage examples completed successfully.")
