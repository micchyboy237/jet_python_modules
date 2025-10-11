import logging
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_visualize_heatmap():
    """Demonstrate visualizing topic similarity heatmap."""
    logging.info("Starting visualize heatmap example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Visualizing topic similarity heatmap...")
    fig = topic_model.visualize_heatmap()
    fig_topics = [int(topic.split("_")[0]) for topic in fig.to_dict()["data"][0]["x"]]
    logging.info(f"Visualized topics: {fig_topics}")
    
    return fig

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_visualize_heatmap()
    logging.info("Heatmap usage examples completed successfully.")
