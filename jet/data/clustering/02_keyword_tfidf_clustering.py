"""
Approach 2 – Keyword‑Based Clustering (TF‑IDF + K‑Means)
Why this approach matters
When interpretability and speed are paramount, TF‑IDF vectorisation remains a strong baseline. It is fully deterministic, requires no GPU, and works exceptionally well when the text contains distinctive, domain‑specific terms (e.g., “cuckold”, “NTR”). This is often the first choice for production‑grade, low‑latency clustering of search results.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# --- 1. Prepare data -------------------------------------------------
def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic clustering for video samples"
    )
    parser.add_argument(
        "samples_path",
        nargs="?",
        default=str(Path(__file__).parent / "mocks" / "02_samples.json"),
        help="Path to samples JSON file",
    )
    return parser.parse_args()


args = parse_args()
samples_path = Path(args.samples_path)
with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)


# --- 1. Basic text cleaning -------------------------------------------
def clean_text(txt):
    txt = txt.lower()
    txt = re.sub(r"[^\w\s]", " ", txt)  # remove punctuation
    txt = re.sub(r"\s+", " ", txt).strip()  # collapse spaces
    return txt


texts = [clean_text(f"{item['code']} {item['text']}") for item in data]

# --- 2. TF-IDF vectorization ------------------------------------------
vectorizer = TfidfVectorizer(
    max_df=0.9,
    min_df=1,
    stop_words="english",
    ngram_range=(1, 2),  # capture phrases like "nude model"
)
tfidf = vectorizer.fit_transform(texts)

# --- 3. Optional: dimensionality reduction (faster, less noise) -------
svd = TruncatedSVD(n_components=50, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf)

# --- 4. K-Means clustering --------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
labels = kmeans.fit_predict(tfidf_reduced)

# --- 5. Extract top keywords per cluster ------------------------------
feature_names = vectorizer.get_feature_names_out()
cluster_keywords = {}
for cid in set(labels):
    centroid = kmeans.cluster_centers_[cid]
    # Map back to original space (approximate)
    orig_centroid = svd.inverse_transform(centroid.reshape(1, -1)).flatten()
    top_idx = orig_centroid.argsort()[-5:][::-1]
    cluster_keywords[cid] = [feature_names[i] for i in top_idx]

# --- 6. Group and display ---------------------------------------------
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    clusters[label].append({"videoId": data[idx]["videoId"], "text": data[idx]["text"]})

for cid, items in clusters.items():
    print(f"\n🔷 Cluster {cid} → Keywords: {cluster_keywords[cid]}")
    for item in items:
        print(f"   - {item['videoId']}: {item['text'][:60]}...")
