import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_visualize_barchart():
    """Demonstrate visualizing topic bar charts."""
    logging.info("Starting visualize barchart example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Visualizing default bar chart...")
    fig = topic_model.visualize_barchart()
    logging.info(f"Number of annotations in default bar chart: {len(fig.to_dict()['layout']['annotations'])}")
    
    logging.info("Visualizing bar chart with top 5 topics...")
    fig_top_n = topic_model.visualize_barchart(top_n_topics=5)
    logging.info(f"Number of annotations in top 5 bar chart: {len(fig_top_n.to_dict()['layout']['annotations'])}")
    
    logging.info("Adding outliers and visualizing bar chart...")
    topic_model.topic_sizes_[-1] = 4
    fig_with_outliers = topic_model.visualize_barchart()
    logging.info(f"Number of annotations with outliers: {len(fig_with_outliers.to_dict()['layout']['annotations'])}")
    
    return fig, fig_top_n

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_visualize_barchart()
    logging.info("Bar usage examples completed successfully.")
