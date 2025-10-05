import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_approximate_distribution():
    """Demonstrate approximating topic distributions for documents."""
    logging.info("Starting approximate distribution example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Approximating topic distribution with padding...")
    topic_distr, _ = topic_model.approximate_distribution(documents, padding=True, batch_size=50)
    logging.info(f"Topic distribution shape: {topic_distr.shape}")
    
    logging.info("Approximating topic distribution without padding...")
    topic_distr_no_padding, _ = topic_model.approximate_distribution(documents, padding=False, batch_size=None)
    logging.info(f"Topic distribution shape (no padding): {topic_distr_no_padding.shape}")
    
    logging.info("Visualizing distribution for first three documents...")
    for i in range(3):
        topic_model.visualize_distribution(topic_distr[i])
    
    logging.info("Approximating topic distribution with token calculation...")
    topic_distr_tokens, topic_token_distr = topic_model.approximate_distribution(documents[:100], calculate_tokens=True)
    logging.info(f"Topic distribution shape (with tokens): {topic_distr_tokens.shape}")
    logging.info(f"Number of token distributions: {len(topic_token_distr)}")
    
    return topic_distr, topic_token_distr

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_approximate_distribution()
    logging.info("Approximate usage examples completed successfully.")
