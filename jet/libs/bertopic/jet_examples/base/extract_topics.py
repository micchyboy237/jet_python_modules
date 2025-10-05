from typing import List
from bertopic import BERTopic
from jet.file.utils import load_file, save_file
from jet.wordnet.topics.topic_parser import configure_topic_model, extract_topics, TopicInfo
import numpy as np
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    print("Using sample documents...")
    # docs: List[str] = [
    #     "The stock market crashed today as tech stocks took a hit.",
    #     "A new study shows the health benefits of a Mediterranean diet.",
    #     "NASA plans to launch a new satellite to monitor climate change.",
    #     "Python is a popular programming language for data science.",
    #     "The local team won the championship after a thrilling final."
    # ]
    chunks = load_file("/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/search/llama_cpp/generated/run_vector_search/chunked_128_32/chunks.json")
    docs: List[str] = [chunk["content"] for chunk in chunks]
    
    print("Fitting BERTopic model...")
    try:
        topic_model: BERTopic = configure_topic_model()
        topics: List[int]
        probs: List[np.ndarray]
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        print(f"Error fitting model: {e}")
        exit(1)
    
    print("Extracting topics...")
    topic_data: List[TopicInfo] = extract_topics(
        topic_model, docs, topics, probs, num_top_words=5, num_representative_docs=3
    )
    
    print("\nExtracted Topics:")
    for topic in topic_data:
        print(f"\nTopic {topic['topic_id']}: {topic['topic_name']}")
        print(f"Document Count: {topic['document_count']}")
        print(f"Top Words: {', '.join(topic['top_words'])}")
        print("Representative Documents:")
        for doc, prob in zip(topic['representative_docs'], topic['doc_probabilities']):
            truncated_doc = f"{doc[:80]}{'...' if len(doc) > 80 else ''}"
            print(f"  - {truncated_doc} (Probability: {prob:.4f})")

    save_file(topic_data, f"{OUTPUT_DIR}/topics.json")
