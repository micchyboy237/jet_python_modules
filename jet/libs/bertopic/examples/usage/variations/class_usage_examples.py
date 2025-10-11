import logging
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups with class labels."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    classes = [newsgroups.target_names[i] for i in newsgroups.target][:1000]
    return documents, classes

def example_topics_per_class():
    """Demonstrate analyzing topics per class with global and local tuning."""
    logging.info("Starting topics per class example...")
    documents, classes = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Computing topics per class with global tuning...")
    topics_per_class_global = topic_model.topics_per_class(documents, classes=classes, global_tuning=True)
    logging.info(f"Global tuning frequency sum: {topics_per_class_global.Frequency.sum()}")
    
    logging.info("Computing topics per class with local tuning...")
    topics_per_class_local = topic_model.topics_per_class(documents, classes=classes, global_tuning=False)
    logging.info(f"Local tuning frequency sum: {topics_per_class_local.Frequency.sum()}")
    
    return topics_per_class_global, topics_per_class_local

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_topics_per_class()
    logging.info("Class usage examples completed successfully.")
