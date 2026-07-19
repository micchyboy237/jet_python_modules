from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()

from typing import List

import numpy as np
import pandas as pd
from jet.adapters.bertopic import BERTopic
from jet.wordnet.topics.topic_parser import configure_topic_model, create_topic_df

if __name__ == "__main__":
    # Sample documents
    print("Using sample documents...")
    docs: List[str] = [
        "The deployment of smart grid tech optimizes wind and solar storage distribution.",
        "Renewable energy grids rely heavily on predictive machine learning algorithms.",
        "Battery backup power management cuts emissions in modernized electrical grids.",
        "Doctors are utilizing machine learning to predict patient heart disease risks.",
        "Advanced clinical diagnostics tools are transforming modern healthcare systems.",
        "Medical AI algorithms assist radiologists in finding early-stage lung tumors.",
        "Central banks are adjusting interest rates to combat rising global inflation.",
        "Stock market volatility increased following the federal reserve interest hike.",
        "Macroeconomic indicators suggest retail consumer spending is slowing down.",
    ]

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
    print(topic_info[["Topic", "Name", "Count"]].head(10).to_string(index=False))

    # Show a sample of documents with their assigned topics and probabilities
    sample_df: pd.DataFrame = create_topic_df(docs, topics, probs)
    print("\nSample Documents, Their Assigned Topics, and Probabilities:")
    print(sample_df.to_string(index=False, max_colwidth=60))
