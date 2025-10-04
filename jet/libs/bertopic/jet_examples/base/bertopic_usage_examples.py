import logging
from tqdm import tqdm
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Configure logging for detailed output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample dataset from 20 newsgroups for topic modeling."""
    logger.info("Loading 20 newsgroups dataset...")
    newsgroups = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups.data[:1000]  # Limit to 1000 documents for example
    timestamps = [i % 10 for i in range(len(documents))]  # Synthetic timestamps
    return documents, timestamps

def example_base_topic_modeling():
    """Demonstrate basic topic modeling with BERTopic."""
    logger.info("Starting basic topic modeling example...")
    documents, _ = load_sample_data()
    
    # Initialize BERTopic model with default settings
    topic_model = BERTopic()
    
    # Fit model with progress tracking
    logger.info("Fitting topic model...")
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        topics, _ = topic_model.fit_transform(documents)
        pbar.update(len(documents))
    
    # Log topic information
    topic_info = topic_model.get_topic_info()
    logger.info(f"Found {len(topic_info)} topics")
    for topic in topic_info['Topic']:
        if topic != -1:  # Skip outliers
            words = topic_model.get_topic(topic)[:5]
            logger.info(f"Topic {topic}: {', '.join([word[0] for word in words])}")
    
    # Save and load model
    logger.info("Saving model...")
    topic_model.save("model_dir", serialization="pytorch", save_ctfidf=True)
    logger.info("Loading model...")
    loaded_model = BERTopic.load("model_dir")
    logger.info("Model loaded successfully")
    
    return topic_model

def example_topic_prediction():
    """Demonstrate predicting topics for new documents."""
    logger.info("Starting topic prediction example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Predict topics for new documents
    new_docs = [
        "The new smartphone features advanced AI capabilities.",
        "Global warming is affecting climate patterns worldwide."
    ]
    logger.info("Predicting topics for new documents...")
    with tqdm(total=len(new_docs), desc="Predicting topics") as pbar:
        topics, probs = topic_model.transform(new_docs)
        pbar.update(len(new_docs))
    
    # Log predictions
    for i, (topic, prob) in enumerate(zip(topics, probs)):
        words = topic_model.get_topic(topic)[:5]
        logger.info(f"Document {i+1} assigned to Topic {topic} (Probability: {prob:.2f}): "
                   f"{', '.join([word[0] for word in words])}")
    
    return topics, probs

def example_topics_over_time():
    """Demonstrate analyzing topics over time."""
    logger.info("Starting topics over time example...")
    documents, timestamps = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Analyze topics over time
    logger.info("Computing topics over time...")
    with tqdm(total=len(timestamps), desc="Analyzing topics over time") as pbar:
        topics_over_time = topic_model.topics_over_time(documents, timestamps)
        pbar.update(len(timestamps))
    
    # Log results
    logger.info("Topics over time analysis complete")
    logger.info(f"Total frequency: {topics_over_time.Frequency.sum()}")
    logger.info(f"Unique topics: {len(topics_over_time.Topic.unique())}")
    
    return topics_over_time

def example_hierarchical_topics():
    """Demonstrate hierarchical topic modeling."""
    logger.info("Starting hierarchical topics example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Compute hierarchical topics
    logger.info("Computing hierarchical topics...")
    with tqdm(total=len(documents), desc="Building hierarchy") as pbar:
        hier_topics = topic_model.hierarchical_topics(documents)
        pbar.update(len(documents))
    
    # Generate and log topic tree
    tree = topic_model.get_topic_tree(hier_topics, tight_layout=False)
    logger.info("Hierarchical topic tree:\n" + tree)
    
    return hier_topics

def example_topic_search():
    """Demonstrate searching for topics similar to a query."""
    logger.info("Starting topic search example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Search for similar topics
    query = "artificial intelligence"
    logger.info(f"Searching for topics similar to: {query}")
    similar_topics, similarity = topic_model.find_topics(query, top_n=2)
    
    # Log results
    for topic, sim in zip(similar_topics, similarity):
        words = topic_model.get_topic(topic)[:5]
        logger.info(f"Topic {topic} (Similarity: {sim:.2f}): "
                   f"{', '.join([word[0] for word in words])}")
    
    return similar_topics, similarity

def example_topic_reduction_and_update():
    """Demonstrate reducing topics and updating topic representations."""
    logger.info("Starting topic reduction and update example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Reduce topics
    original_topics = len(topic_model.get_topic_info())
    nr_topics = max(2, original_topics - 1)
    logger.info(f"Reducing topics from {original_topics} to {nr_topics}...")
    with tqdm(total=len(documents), desc="Reducing topics") as pbar:
        topic_model.reduce_topics(documents, nr_topics=nr_topics)
        pbar.update(len(documents))
    
    # Update topic representations
    logger.info("Updating topic representations with bigrams...")
    original_vectorizer = topic_model.vectorizer_model
    topic_model.update_topics(documents, n_gram_range=(2, 2))
    logger.info("Restoring original vectorizer...")
    topic_model.update_topics(documents, vectorizer_model=original_vectorizer)
    
    # Log results
    logger.info(f"Topics reduced to {len(topic_model.get_topic_info())}")
    
    return topic_model

def example_topic_merging():
    """Demonstrate merging topics."""
    logger.info("Starting topic merging example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    
    # Fit model
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Merge topics
    topics_to_merge = [0, 1]
    logger.info(f"Merging topics {topics_to_merge}...")
    with tqdm(total=len(documents), desc="Merging topics") as pbar:
        topic_model.merge_topics(documents, topics_to_merge)
        pbar.update(len(documents))
    
    # Log results
    logger.info(f"Topics after merging: {len(topic_model.get_topic_info())}")
    
    return topic_model

if __name__ == "__main__":
    logger.info("Running BERTopic usage examples...")
    example_base_topic_modeling()
    example_topic_prediction()
    example_topics_over_time()
    example_hierarchical_topics()
    example_topic_search()
    example_topic_reduction_and_update()
    example_topic_merging()
    logger.info("All examples completed successfully.")
