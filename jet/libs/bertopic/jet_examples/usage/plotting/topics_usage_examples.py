import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_visualize_topics():
    """Demonstrate visualizing topics in a BERTopic model."""
    logging.info("Starting visualize topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Visualizing all topics...")
    fig = topic_model.visualize_topics()
    logging.info("Visualizing top 5 topics...")
    fig_top_n = topic_model.visualize_topics(top_n_topics=5)
    
    logging.info("Adding outliers and visualizing topics...")
    topic_model.topic_sizes_[-1] = 4
    fig_with_outliers = topic_model.visualize_topics()
    logging.info("Visualizing top 5 topics with outliers...")
    fig_top_n_with_outliers = topic_model.visualize_topics(top_n_topics=5)
    
    return fig, fig_top_n

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_visualize_topics()
    logging.info("Topics usage examples completed successfully.")
