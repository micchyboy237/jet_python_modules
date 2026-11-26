"""
Demo 8: Concept Extraction with GLiFormer Vectorizer
Uses GLiFormer to extract structured topics/concepts from text using LLMs.
Requires: pip install turftopic[gliformer]
Note: This demo uses a small subset of data due to the computational cost of LLM-based extraction.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from mock import SAMPLE_CORPUS_LG
from turftopic import KeyNMF
from turftopic.vectorizers.gliformer import ConceptVectorizer

print("Initializing ConceptVectorizer with GLiFormer...")
# GLiFormer extracts concepts like "Stock Market", "Artificial Intelligence"
# instead of single words.
# min_df=1 is used because the dataset is small and concepts are sparse.
vectorizer = ConceptVectorizer(min_df=1)

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model with Concept Vectorizer...")
# Using fewer components for this small, concept-rich dataset
model = KeyNMF(3, vectorizer=vectorizer, encoder=encoder)

# GLiFormer extraction is slower than standard tokenization
model.fit(SAMPLE_CORPUS_LG)

print("\n--- Extracted Concepts/Topics ---")
model.print_topics()
