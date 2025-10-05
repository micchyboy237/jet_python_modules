import numpy as np
import pandas as pd
from bertopic import BERTopic
from typing import List
from jet.file.utils import load_file, save_file
from jet.wordnet.topics.topic_parser import configure_topic_model, create_topic_df
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    # Sample documents
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

    # Fit BERTopic model
    print("Fitting BERTopic model...")
    try:
        topic_model: BERTopic = configure_topic_model()
        topics: List[int]
        probs: List[np.ndarray]
        topics, probs = topic_model.fit_transform(docs)
    except ValueError as e:
        print(f"Error fitting model: {e}")
        exit(1)

    # Get topic info as a DataFrame
    print("Getting topic info...")
    topic_info: pd.DataFrame = topic_model.get_topic_info()

    # Display the first few topics and their details
    print("Top Topics:")
    print(topic_info[['Topic', 'Name', 'Count']].head(
        10).to_string(index=False))

    # Show a sample of documents with their assigned topics and probabilities
    sample_df: pd.DataFrame = create_topic_df(docs, topics, probs)
    print("\nSample Documents, Their Assigned Topics, and Probabilities:")
    print(sample_df.to_string(index=False, max_colwidth=60))

    # Save to separate JSON files
    save_file(topic_info.to_dict(orient="records"), f"{OUTPUT_DIR}/topic_info.json")
    save_file(sample_df.to_dict(orient="records"), f"{OUTPUT_DIR}/sample_documents.json")