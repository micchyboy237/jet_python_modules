import logging
import pandas as pd
from packaging import version
from sklearn import __version__ as sklearn_version
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from jet.adapters.bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer

def load_sample_data():
    """Load sample dataset from 20 newsgroups."""
    logging.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]
    return documents

def example_ctfidf():
    """Demonstrate c-TF-IDF vectorization."""
    logging.info("Starting c-TF-IDF example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    topics = topic_model.topics_
    documents_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    documents_per_topic = documents_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    preprocessed_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    
    logging.info("Applying CountVectorizer...")
    count = topic_model.vectorizer_model.fit(preprocessed_docs)
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = count.get_feature_names_out()
    else:
        words = count.get_feature_names()
    X = count.transform(preprocessed_docs)
    
    logging.info("Applying ClassTfidfTransformer...")
    transformer = ClassTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)
    
    logging.info(f"Number of words: {len(words)}")
    logging.info(f"c-TF-IDF shape: {c_tf_idf.shape}")
    
    return c_tf_idf, words

def example_ctfidf_custom_cv():
    """Demonstrate c-TF-IDF vectorization with custom CountVectorizer."""
    logging.info("Starting c-TF-IDF with custom CountVectorizer example...")
    documents = load_sample_data()
    topic_model = BERTopic()
    topic_model.vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words="english")
    logging.info("Fitting topic model...")
    topic_model.fit(documents)
    
    topics = topic_model.topics_
    documents_df = pd.DataFrame({"Document": documents, "ID": range(len(documents)), "Topic": topics})
    documents_per_topic = documents_df.groupby(["Topic"], as_index=False).agg({"Document": " ".join})
    preprocessed_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
    
    logging.info("Applying custom CountVectorizer...")
    count = topic_model.vectorizer_model.fit(preprocessed_docs)
    if version.parse(sklearn_version) >= version.parse("1.0.0"):
        words = count.get_feature_names_out()
    else:
        words = count.get_feature_names()
    X = count.transform(preprocessed_docs)
    
    logging.info("Applying ClassTfidfTransformer...")
    transformer = ClassTfidfTransformer().fit(X)
    c_tf_idf = transformer.transform(X)
    
    logging.info(f"Number of words: {len(words)}")
    logging.info(f"c-TF-IDF shape: {c_tf_idf.shape}")
    
    return c_tf_idf, words

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_ctfidf()
    example_ctfidf_custom_cv()
    logging.info("c-TF-IDF usage examples completed successfully.")
