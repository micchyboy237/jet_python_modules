"""
Demo 3: Automated Topic Naming with LLMs
Uses Jet's llama.cpp adapter for both embeddings and LLM naming.
"""

from jet.adapters.turftopic.chat import LlamaCppAnalyzer
from jet.adapters.turftopic.embed import LlamaCppEncoder
from shared_samples import SAMPLE_CORPUS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model...")
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(3, vectorizer=vectorizer, encoder=encoder).fit(SAMPLE_CORPUS)

print("Initializing Jet Llama Analyzer...")
namer = LlamaCppAnalyzer()

print("Generating topic names...")
model.rename_topics(namer)

print("\n--- Named Topics ---")
model.print_topics()
