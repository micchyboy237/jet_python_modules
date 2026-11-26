"""
Demo 1: Basic Contextual Topic Modeling with KeyNMF
Uses Jet's llama.cpp adapter for embeddings.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from mock import SAMPLE_CORPUS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model...")
# Initialize KeyNMF with 3 topics for this small dataset
# Use min_df=1 because the dataset is small (default is 10)
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(n_components=3, top_n=5, vectorizer=vectorizer, encoder=encoder)

# Fit the model
document_topic_matrix = model.fit_transform(SAMPLE_CORPUS)

print("\n--- Top Topics ---")
model.print_topics()

print("\n--- Representative Documents for Topic 0 ---")
model.print_representative_documents(0, SAMPLE_CORPUS, document_topic_matrix)
