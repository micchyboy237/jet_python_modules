from typing import List
from bertopic import BERTopic, TopicMapper
import pandas as pd

# Sample documents for demonstration (consistent with previous artifacts)
SAMPLE_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing",
    "BERTopic is a powerful topic modeling tool",
    "Natural language models are improving rapidly",
    "The dog sleeps while the fox runs"
]

def example_topicmapper_add_new_topics() -> None:
    """
    Demonstrates the add_new_topics method of TopicMapper.
    - Adds new topic assignments to the TopicMapper and updates the BERTopic model.
    - Uses SAMPLE_DOCS to fit the model and generate initial topics.
    - Parameters:
        - documents: DataFrame with new documents to assign to topics.
        - topics: List of new topic assignments.
        - topic_model: BERTopic model to update with new topics.
    """
    # Initialize and fit BERTopic model with SAMPLE_DOCS
    model = BERTopic(min_topic_size=2)
    model.fit(documents=SAMPLE_DOCS)
    
    # Create TopicMapper with initial topics
    topic_mapper = TopicMapper(topics=model.topics_)
    
    # Prepare new documents and topic assignments
    new_docs = pd.DataFrame({"Document": ["New doc about foxes", "New tech topic"]})
    new_topics = [0, 1]  # Assign new docs to existing topics 0 and 1
    
    # Add new topics to TopicMapper and update model
    topic_mapper.add_new_topics(
        documents=new_docs,
        topics=new_topics,
        topic_model=model
    )
    
    # Print updated topics to show the effect
    print(f"Updated topics in model after add_new_topics: {model.topics_}")