import logging
from scipy.cluster import hierarchy as sch
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_hierarchical_topics():
    """Demonstrate hierarchical topic modeling."""
    logging.info("Starting hierarchical topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Computing hierarchical topics...")
    hierarchical_topics = topic_model.hierarchical_topics(documents)
    logging.info(f"Number of hierarchical topics: {len(hierarchical_topics)}")
    
    return hierarchical_topics

def example_hierarchical_topics_with_linkage():
    """Demonstrate hierarchical topic modeling with custom linkage function."""
    logging.info("Starting hierarchical topics with linkage example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Computing hierarchical topics with custom linkage...")
    linkage_function = lambda x: sch.linkage(x, "single", optimal_ordering=True)
    hierarchical_topics = topic_model.hierarchical_topics(documents, linkage_function=linkage_function)
    tree = topic_model.get_topic_tree(hierarchical_topics)
    logging.info(f"Topic tree:\n{tree}")
    
    return hierarchical_topics, tree

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_hierarchical_topics()
    example_hierarchical_topics_with_linkage()
    logging.info("Hierarchy usage examples completed successfully.")
