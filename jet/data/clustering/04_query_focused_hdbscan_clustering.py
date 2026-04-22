"""
Approach 4 – Query‑Focused Clustering with HDBSCAN

Why this approach matters
When you have a specific user query (e.g., “find all cuckold‑themed videos”), you can pre‑cluster the corpus and then route the query to the most relevant clusters. This reduces noise and boosts precision by only searching semantically coherent groups. HDBSCAN is especially suitable because it automatically determines the number of clusters and identifies outliers (videos that don‘t match the query).
"""

import json
from pathlib import Path

import hdbscan
from jet.adapters.llama_cpp.embed_utils import embed
from sentence_transformers import util

user_query = "videos about cuckold and wife swapping"
query_prefix = "search_query: "
doc_prefix = "search_document: "


# --- 1. Prepare data -------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic clustering for video samples"
    )
    parser.add_argument(
        "samples_path",
        nargs="?",
        default=str(Path(__file__).parent / "mocks" / "04_samples.json"),
        help="Path to samples JSON file",
    )
    return parser.parse_args()


args = parse_args()
samples_path = Path(args.samples_path)
with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# --- 1. Embeddings for corpus and query -------------------------------
texts = [f"{doc_prefix}{item['code']} {item['text']}" for item in data]
all_embeddings = embed([f"{query_prefix}{user_query}"] + texts)
query_embedding = all_embeddings[0]
corpus_embeddings = all_embeddings[1:]

# --- 2. HDBSCAN clustering (density‑based, outlier‑aware) ------------
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2, metric="euclidean", cluster_selection_epsilon=0.5
)
labels = clusterer.fit_predict(corpus_embeddings)

# --- 3. Find which cluster(s) are most relevant to the query ---------
unique_labels = set(labels) - {-1}  # exclude noise
cluster_centroids = {}
for label in unique_labels:
    mask = labels == label
    cluster_centroids[label] = corpus_embeddings[mask].mean(axis=0)

# Compute cosine similarity between query and each cluster centroid
best_cluster = None
best_sim = -1
for label, centroid in cluster_centroids.items():
    sim = util.cos_sim(query_embedding, centroid).item()
    if sim > best_sim:
        best_sim = sim
        best_cluster = label

# --- 4. Retrieve items from the most relevant cluster -----------------
relevant_items = []
for idx, label in enumerate(labels):
    if label == best_cluster:
        # Compute direct similarity to query (optional ranking)
        sim_to_query = util.cos_sim(query_embedding, corpus_embeddings[idx]).item()
        relevant_items.append({"item": data[idx], "score": sim_to_query})

# Sort by relevance to query
relevant_items.sort(key=lambda x: x["score"], reverse=True)

# --- 5. Display -------------------------------------------------------
print(f"🔍 Query: '{user_query}'")
print(f"🎯 Matched cluster: {best_cluster} (similarity: {best_sim:.3f})")
print("\n📋 Results:")
for r in relevant_items:
    it = r["item"]
    print(f"   - [{r['score']:.3f}] {it['videoId']}: {it['text'][:60]}...")
