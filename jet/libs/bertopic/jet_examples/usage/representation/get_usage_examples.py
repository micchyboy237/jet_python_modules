import logging
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_get_topic():
    """Demonstrate retrieving topics for a BERTopic model."""
    logging.info("Starting get topic example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topics...")
    topics = set(topic_model.topics_)
    for topic in topics:
        topic_words = topic_model.get_topic(topic)
        logging.info(f"Topic {topic}: {', '.join([word[0] for word in topic_words[:5]])}")
    
    unknown_topic = topic_model.get_topic(500)
    logging.info(f"Unknown topic (500) result: {unknown_topic}")
    
    return topics

def example_get_topics():
    """Demonstrate retrieving all topic representations."""
    logging.info("Starting get topics example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving all topics...")
    topics = topic_model.get_topics()
    for topic_id, words in topics.items():
        logging.info(f"Topic {topic_id}: {', '.join([word[0] for word in words[:5]])}")
    
    return topics

def example_get_topic_freq():
    """Demonstrate retrieving topic frequencies."""
    logging.info("Starting get topic frequency example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topic frequencies...")
    topic_freq = topic_model.get_topic_freq()
    logging.info(f"Topic frequency DataFrame:\n{topic_freq.head()}")
    
    for topic in set(topic_model.topics_):
        freq = topic_model.get_topic_freq(topic)
        logging.info(f"Frequency for topic {topic}: {freq}")
    
    return topic_freq

def example_get_representative_docs():
    """Demonstrate retrieving representative documents for topics."""
    logging.info("Starting get representative documents example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving representative documents...")
    all_docs = topic_model.get_representative_docs()
    for topic_id, docs in all_docs.items():
        logging.info(f"Topic {topic_id} representative docs: {docs}")
    
    return all_docs

def example_get_topic_info():
    """Demonstrate retrieving topic information."""
    logging.info("Starting get topic info example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Retrieving topic information...")
    topic_info = topic_model.get_topic_info()
    logging.info(f"Topic info DataFrame:\n{topic_info.head()}")
    
    for topic in set(topic_model.topics_):
        info = topic_model.get_topic_info(topic)
        logging.info(f"Info for topic {topic}:\n{info}")
    
    return topic_info

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_get_topic()
    example_get_topics()
    example_get_topic_freq()
    example_get_representative_docs()
    example_get_topic_info()
    logging.info("Get usage examples completed successfully.")
