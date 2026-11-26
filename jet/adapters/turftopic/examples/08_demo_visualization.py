"""
Demo 8: Visualization with TopicWizard
Launches an interactive web app to explore the model.
Requires: pip install turftopic[topic-wizard]
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from mock import SAMPLE_CORPUS_LG
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model...")
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(3, vectorizer=vectorizer, encoder=encoder)
model.fit(SAMPLE_CORPUS_LG)

print("\nLaunching TopicWizard...")
print("Check your browser for the interactive visualization.")
try:
    import topicwizard

    topicwizard.visualize(SAMPLE_CORPUS_LG, model=model)
except ImportError:
    print("topicwizard not installed. Run: pip install turftopic[topic-wizard]")
except Exception as e:
    print(f"Could not launch TopicWizard: {e}")
