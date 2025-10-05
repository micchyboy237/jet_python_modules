import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_visualize_term_rank():
    """Demonstrate visualizing term rank for topics."""
    logging.info("Starting visualize term rank example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Visualizing term rank...")
    topic_model.visualize_term_rank()
    logging.info("Term rank visualization completed")
    
    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_visualize_term_rank()
    logging.info("Term rank usage examples completed successfully.")
