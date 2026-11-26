"""
Demo 7: Multimodal Topic Modeling
Demonstrates the API usage with synthetic image placeholders.
Note: Actual image processing requires a multimodal encoder in Jet/Llama.cpp.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from mock import SAMPLE_CORPUS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

# Placeholder for images (In real use, these would be PIL Images or paths)
# For this demo, we use None to show the API structure
images = [None] * len(SAMPLE_CORPUS)

print("Initializing LlamaCppEncoder...")
# Note: Standard LlamaCppEncoder might not support images directly unless configured
# with a multimodal model in config.py. This demo shows the API structure.
encoder = LlamaCppEncoder()

print("Initializing Multimodal KeyNMF...")
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(2, vectorizer=vectorizer, encoder=encoder)

try:
    # In a real scenario with a multimodal encoder configured in Jet:
    # doc_topic_matrix = model.fit_transform_multimodal(SAMPLE_CORPUS, images)

    print("Multimodal API ready.")
    print("To run with actual images:")
    print("  1. Configure a multimodal encoder in jet/adapters/llama_cpp/config.py")
    print("  2. Pass valid PIL Images or paths to the 'images' list")
    print("  3. Call model.fit_transform_multimodal(corpus, images)")

except Exception as e:
    print(f"Error setting up multimodal demo: {e}")
