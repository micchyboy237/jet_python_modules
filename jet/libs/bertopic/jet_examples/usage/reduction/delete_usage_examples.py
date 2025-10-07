import logging
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_delete_topics():
    """Demonstrate deleting topics from a BERTopic model."""
    logging.info("Starting delete topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Deleting topics [1, 2]...")
    nr_topics_before = len(set(topic_model.topics_))
    topics_to_delete = [1, 2]
    topic_model.delete_topics(topics_to_delete)
    logging.info(f"Number of topics reduced from {nr_topics_before} to {len(set(topic_model.topics_))}")    
    
    logging.info("Deleting additional topics...")
    remaining_topics = sorted(list(set(topic_model.topics_)))
    remaining_topics = [t for t in remaining_topics if t != -1]
    topics_to_delete = remaining_topics[:2]
    topic_model.delete_topics(topics_to_delete)
    logging.info(f"Number of topics further reduced to {len(set(topic_model.topics_))}")    
    
    return topic_model

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_delete_topics()
    logging.info("Delete usage examples completed successfully.")
