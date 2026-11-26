"""
Demo 4: Phrase-based Vectorization with SpaCy
Note: This demo uses SpaCy for phrase extraction but still uses Jet for embeddings.
Requires: pip install spacy && python -m spacy download en_core_web_sm
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from shared_samples import SAMPLE_CORPUS
from turftopic import KeyNMF
from turftopic.vectorizers.spacy import NounPhraseCountVectorizer

print("Initializing Noun Phrase Vectorizer...")
# SpaCy vectorizers usually handle small datasets better, but we ensure no strict filtering
vectorizer = NounPhraseCountVectorizer("en_core_web_sm")

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF with Phrase Vectorizer...")
model = KeyNMF(3, vectorizer=vectorizer, encoder=encoder)
model.fit(SAMPLE_CORPUS)

print("\n--- Phrase-based Topics ---")
model.print_topics()
