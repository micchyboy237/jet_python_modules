import logging
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_online_cv():
    """Demonstrate online CountVectorizer with BERTopic."""
    logging.info("Starting online CountVectorizer example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Updating topics with OnlineCountVectorizer...")
    original_topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]
    vectorizer_model = OnlineCountVectorizer(stop_words="english", ngram_range=(2, 2))
    topic_model.update_topics(documents, vectorizer_model=vectorizer_model)
    new_topics = [topic_model.get_topic(topic) for topic in set(topic_model.topics_)]
    
    for old_topic, new_topic in zip(original_topics, new_topics):
        if old_topic[0][0] != "":
            logging.info(f"Old topic: {old_topic[:3]}, New topic: {new_topic[:3]}")
    
    return new_topics

def example_clean_bow():
    """Demonstrate cleaning Bag-of-Words with OnlineCountVectorizer."""
    logging.info("Starting clean Bag-of-Words example...")
    documents = load_sample_data()
    topic_model = BERTopic(vectorizer_model=OnlineCountVectorizer(stop_words="english"))
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    logging.info("Cleaning Bag-of-Words...")
    original_shape = topic_model.vectorizer_model.X_.shape
    topic_model.vectorizer_model.delete_min_df = 2
    topic_model.vectorizer_model._clean_bow()
    new_shape = topic_model.vectorizer_model.X_.shape
    logging.info(f"Original shape: {original_shape}, New shape: {new_shape}")
    
    return new_shape

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_online_cv()
    example_clean_bow()
    logging.info("Online CountVectorizer usage examples completed successfully.")
