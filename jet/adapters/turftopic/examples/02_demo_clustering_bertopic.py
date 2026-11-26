"""
Demo 2: Clustering-based Topic Modeling (BERTopic style)
Uses Jet's llama.cpp adapter for embeddings.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from shared_samples import SAMPLE_CORPUS
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import BERTopic

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting BERTopic model...")
# BERTopic uses UMAP and HDBSCAN by default
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")

# Configure HDBSCAN with a smaller min_cluster_size for our small dataset (15 docs)
clustering = HDBSCAN(min_cluster_size=2, min_samples=1)

model = BERTopic(
    n_reduce_to=3, vectorizer=vectorizer, encoder=encoder, clustering=clustering
)

# Fit the model
document_topic_matrix = model.fit_transform(SAMPLE_CORPUS)

print("\n--- Top Topics ---")
model.print_topics()
