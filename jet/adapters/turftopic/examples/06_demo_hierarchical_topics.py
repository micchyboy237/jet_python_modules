"""
Demo 6: Hierarchical Topic Modeling
Discovers sub-topics and prints the hierarchy tree.
"""

from jet.adapters.turftopic.embed import LlamaCppEncoder
from shared_samples import SAMPLE_CORPUS
from sklearn.feature_extraction.text import CountVectorizer
from turftopic import KeyNMF

print("Initializing LlamaCppEncoder...")
encoder = LlamaCppEncoder()

print("Fitting KeyNMF model...")
# Use min_df=1 because the dataset is small
vectorizer = CountVectorizer(min_df=1, stop_words="english")
model = KeyNMF(
    2, vectorizer=vectorizer, encoder=encoder
)  # Start with fewer topics to allow division
model.fit(SAMPLE_CORPUS)

print("\n--- Topic Hierarchy ---")
if hasattr(model, "hierarchy") and model.hierarchy:
    # Divide the first topic into 2 subtopics
    root = model.hierarchy
    if root.children:
        first_child = root.children[0]
        if hasattr(first_child, "divide"):
            first_child.divide(n_subtopics=2)

    # Print the tree structure
    model.hierarchy.print_tree(top_k=5, max_depth=2)
else:
    print("No hierarchy found.")
