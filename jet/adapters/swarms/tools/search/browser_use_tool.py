from pathlib import Path
from jet.file.utils import save_file

# Define output directory as a constant
OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem

def save_topic_info(topic_model: BERTopic, output_path: str) -> None:
    """Save topic information to a JSON file."""
    topic_info = topic_model.get_topic_info().to_dict(orient="records")
    formatted_data = [
        {
            "topic_id": row["Topic"],
            "count": row["Count"],
            "name": row["Name"],
            "top_words": [word[0] for word in topic_model.get_topic(row["Topic"])[:5]]
        }
        for row in topic_info if row["Topic"] != -1
    ]
    save_file(formatted_data, output_path, verbose=True)

def example_base_topic_modeling():
    """Demonstrate basic topic modeling with BERTopic."""
    logger.info("Starting basic topic modeling example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    logger.info("Fitting topic model...")
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        topics, _ = topic_model.fit_transform(documents)
        pbar.update(len(documents))
    
    # Save topic information
    output_path = OUTPUT_DIR / "base_topic_modeling.json"
    save_topic_info(topic_model, output_path)
    
    # Log topic information
    topic_info = topic_model.get_topic_info()
    logger.info(f"Found {len(topic_info)} topics")
    for topic in topic_info['Topic']:
        if topic != -1:
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
    
    # Save prediction results
    output_path = OUTPUT_DIR / "topic_predictions.jsonl"
    predictions = [
        {
            "document": doc,
            "topic_id": topic,
            "probability": float(prob),
            "top_words": [word[0] for word in topic_model.get_topic(topic)[:5]]
        }
        for doc, topic, prob in zip(new_docs, topics, probs)
    ]
    save_file(predictions, output_path, verbose=True, append=False)
    
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
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Analyze topics over time
    logger.info("Computing topics over time...")
    with tqdm(total=len(timestamps), desc="Analyzing topics over time") as pbar:
        topics_over_time = topic_model.topics_over_time(documents, timestamps)
        pbar.update(len(timestamps))
    
    # Save topics over time
    output_path = OUTPUT_DIR / "topics_over_time.json"
    formatted_data = topics_over_time.to_dict(orient="records")
    for row in formatted_data:
        row["top_words"] = [word[0] for word in topic_model.get_topic(row["Topic"])[:5]]
    save_file(formatted_data, output_path, verbose=True)
    
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
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Compute hierarchical topics
    logger.info("Computing hierarchical topics...")
    with tqdm(total=len(documents), desc="Building hierarchy") as pbar:
        hier_topics = topic_model.hierarchical_topics(documents)
        pbar.update(len(documents))
    
    # Save hierarchical topics and tree
    output_path = OUTPUT_DIR / "hierarchical_topics.json"
    formatted_data = hier_topics.to_dict(orient="records")
    for row in formatted_data:
        row["top_words"] = [word[0] for word in topic_model.get_topic(row["Topic"])[:5]]
    save_file(formatted_data, output_path, verbose=True)
    
    output_tree_path = OUTPUT_DIR / "hierarchical_topic_tree.txt"
    tree = topic_model.get_topic_tree(hier_topics, tight_layout=False)
    save_file(tree, output_tree_path, verbose=True)
    
    # Log topic tree
    logger.info("Hierarchical topic tree:\n" + tree)
    
    return hier_topics

def example_topic_search():
    """Demonstrate searching for topics similar to a query."""
    logger.info("Starting topic search example...")
    documents, _ = load_sample_data()
    topic_model = BERTopic()
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Search for similar topics
    query = "artificial intelligence"
    logger.info(f"Searching for topics similar to: {query}")
    similar_topics, similarity = topic_model.find_topics(query, top_n=2)
    
    # Save search results
    output_path = OUTPUT_DIR / "topic_search.json"
    search_results = [
        {
            "query": query,
            "topic_id": topic,
            "similarity": float(sim),
            "top_words": [word[0] for word in topic_model.get_topic(topic)[:5]]
        }
        for topic, sim in zip(similar_topics, similarity)
    ]
    save_file(search_results, output_path, verbose=True)
    
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
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Reduce topics
    original_topics = len(topic_model.get_topic_info())
    nr_topics = max(2, original_topics - 1)
    logger.info(f"Reducing topics from {original_topics} to {nr_topics}...")
    with tqdm(total=len(documents), desc="Reducing topics") as pbar:
        topic_model.reduce_topics(documents, nr_topics=nr_topics)
        pbar.update(len(documents))
    
    # Save reduced topics
    output_path = OUTPUT_DIR / "reduced_topics.json"
    save_topic_info(topic_model, output_path)
    
    # Update topic representations
    logger.info("Updating topic representations with bigrams...")
    original_vectorizer = topic_model.vectorizer_model
    topic_model.update_topics(documents, n_gram_range=(2, 2))
    
    # Save bigram topics
    output_bigrams_path = OUTPUT_DIR / "bigram_topics.json"
    save_topic_info(topic_model, output_bigrams_path)
    
    # Restore original vectorizer
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
    logger.info("Fitting topic model...")
    topic_model.fit(documents)
    
    # Merge topics
    topics_to_merge = [0, 1]
    logger.info(f"Merging topics {topics_to_merge}...")
    with tqdm(total=len(documents), desc="Merging topics") as pbar:
        topic_model.merge_topics(documents, topics_to_merge)
        pbar.update(len(documents))
    
    # Save merged topics
    output_path = OUTPUT_DIR / "merged_topics.json"
    save_topic_info(topic_model, output_path)
    
    # Log results
    logger.info(f"Topics after merging: {len(topic_model.get_topic_info())}")
    
    return topic_model