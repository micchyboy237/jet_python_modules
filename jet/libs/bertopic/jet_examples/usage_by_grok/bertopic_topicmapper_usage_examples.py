from typing import List, Mapping
import pandas as pd
from bertopic import BERTopic, TopicMapper

# Sample documents for demonstration (consistent with previous artifacts)
SAMPLE_DOCS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is transforming natural language processing",
    "BERTopic is a powerful topic modeling tool",
    "Natural language models are improving rapidly",
    "The dog sleeps while the fox runs"
]

def example_topicmapper_init() -> TopicMapper:
    """
    Demonstrates the initialization of the TopicMapper class.
    - Creates a TopicMapper with topic assignments from a fitted BERTopic model.
    - Uses SAMPLE_DOCS to fit the model and generate topics.
    """
    model = BERTopic(min_topic_size=2)
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    print(f"TopicMapper initialized with topics: {topic_mapper.topics}")
    return topic_mapper

def example_topicmapper_get_mappings() -> Mapping[int, int]:
    """
    Demonstrates the get_mappings method of TopicMapper.
    - Retrieves the mapping of original to updated topic IDs after fitting a model.
    - Uses SAMPLE_DOCS to fit the BERTopic model and initialize TopicMapper.
    """
    model = BERTopic(min_topic_size=2)
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    mappings = topic_mapper.get_mappings(original_topics=True)
    print(f"Topic mappings: {mappings}")
    return mappings

def example_topicmapper_add_mappings() -> None:
    """
    Demonstrates the add_mappings method of TopicMapper.
    - Adds new topic mappings and updates the BERTopic model.
    - Uses SAMPLE_DOCS to fit the model and demonstrate mapping updates.
    """
    model = BERTopic(min_topic_size=2)
    model.fit(documents=SAMPLE_DOCS)
    topic_mapper = TopicMapper(topics=model.topics_)
    new_mappings = {0: 1}  # Example: Map topic 0 to topic 1
    topic_mapper.add_mappings(mappings=new_mappings, topic_model=model)
    print(f"Updated topics in model: {model.topics_}")
    # Note: This updates the model's topics_, so we print them to show the effect