"""
Demo 5: Dynamic Topic Modeling
Analyzes how topics evolve over time using timestamps.
"""

from datetime import datetime

import numpy as np
from jet.adapters.turftopic.embed import LlamaCppEncoder
from shared_samples import SAMPLE_CORPUS, SAMPLE_TIMESTAMPS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

# Convert string timestamps to datetime objects
timestamps = [datetime.strptime(ts, "%Y-%m-%d") for ts in SAMPLE_TIMESTAMPS]

print("Fitting Dynamic KeyNMF model...")
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(3, vectorizer=vectorizer, encoder=encoder)

# bins=3 creates 3 time slices (Jan, Feb, Mar)
doc_topic_matrix = model.fit_transform_dynamic(
    raw_documents=SAMPLE_CORPUS, timestamps=timestamps, bins=3
)

print("\n--- Temporal Topic Evolution ---")
vocab = model.get_vocab()
# Print top words for each time slice for the first topic
for t_idx, time_slice in enumerate(model.temporal_components_):
    if time_slice.shape[0] > 0:
        top_indices = np.argsort(-time_slice[0])[:5]
        top_words = [vocab[i] for i in top_indices if i < len(vocab)]
        print(f"Time Slice {t_idx}: {top_words}")
