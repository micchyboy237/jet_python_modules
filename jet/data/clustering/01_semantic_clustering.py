"""
Embedding‑Based Clustering (BERT + K‑Means)

Why this approach matters
Modern sentence‑level embeddings (e.g., from all‑MiniLM‑L6‑v2) capture deep semantic meaning, not just word overlap. When combined with K‑Means, they produce high‑quality, interpretable clusters even for short, noisy text. This is the most widely adopted technique in 2025/2026 for short‑text clustering
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from jet.adapters.llama_cpp.embed_utils import embed
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

doc_prefix = "search_document: "

# --- 1. Prepare data -------------------------------------------------
samples_path = Path(__file__).parent / "mocks" / "01_samples.json"
with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Concatenate relevant text fields for each video
texts = [f"{doc_prefix}{item['code']} {item['text']}" for item in data]

# --- 2. Generate embeddings ------------------------------------------
embeddings = embed(texts)

# --- 3. Cluster with K-Means -----------------------------------------
n_clusters = 2  # adjust based on data size; use silhouette/elbow for tuning
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
cluster_labels = kmeans.fit_predict(embeddings)

# --- 4. Extract descriptive keywords per cluster ---------------------
vectorizer = TfidfVectorizer(max_features=5, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(texts)

# Get top keywords for each cluster
cluster_keywords = {}
for cluster_id in range(n_clusters):
    mask = cluster_labels == cluster_id
    if not np.any(mask):
        continue
    cluster_tfidf = tfidf_matrix[mask].mean(axis=0)
    top_indices = np.asarray(cluster_tfidf).flatten().argsort()[-5:][::-1]
    keywords = [vectorizer.get_feature_names_out()[i] for i in top_indices]
    cluster_keywords[cluster_id] = keywords

# --- 5. Group results -------------------------------------------------
clusters = defaultdict(list)
for idx, label in enumerate(cluster_labels):
    clusters[int(label)].append(
        {
            "videoId": data[idx]["videoId"],
            "text": data[idx]["text"],
            "url": data[idx]["url"],
        }
    )

# --- 6. Display -------------------------------------------------------
for cid, items in clusters.items():
    print(f"\n🔷 Cluster {cid} → Keywords: {', '.join(cluster_keywords[cid])}")
    for item in items:
        print(f"   - {item['videoId']}: {item['text'][:60]}...")
