import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups with timestamps."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    timestamps = [i % 10 for i in range(len(documents))]
    return documents, timestamps

def example_topics_over_time():
    """Demonstrate analyzing topics over time."""
    logging.info("Starting topics over time example...")
    documents, timestamps = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Computing topics over time...")
    topics_over_time = topic_model.topics_over_time(documents, timestamps)
    logging.info(f"Total frequency: {topics_over_time.Frequency.sum()}")
    logging.info(f"Unique timestamps: {len(topics_over_time.Timestamp.unique())}")
    
    return topics_over_time

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_topics_over_time()
    logging.info("Dynamic usage examples completed successfully.")
