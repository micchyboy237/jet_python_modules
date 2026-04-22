"""
Approach 3 – Hierarchical Agglomerative Clustering

Why this approach matters
Hierarchical clustering builds a tree (dendrogram) that reveals relationships at multiple granularities. This is ideal when the number of clusters is unknown and you want to explore how items naturally group together. It is also deterministic and requires no iterative tuning.
"""

import json
from collections import defaultdict
from pathlib import Path

from jet.adapters.llama_cpp.embed_utils import embed
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

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
        default=str(Path(__file__).parent / "mocks" / "03_samples.json"),
        help="Path to samples JSON file",
    )
    return parser.parse_args()


args = parse_args()
samples_path = Path(args.samples_path)
with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- 1. Embeddings ----------------------------------------------------
texts = [f"{doc_prefix}{item['code']} {item['text']}" for item in data]
embeddings = embed(texts)

# --- 2. Agglomerative clustering (cosine affinity) --------------------
# Use 'ward' linkage with Euclidean distance, or precomputed cosine distances
distance_matrix = cosine_distances(embeddings)

clustering = AgglomerativeClustering(
    n_clusters=None,  # cut later by threshold
    distance_threshold=0.5,  # cosine distance < 0.5 => similar
    metric="precomputed",
    linkage="average",
)
labels = clustering.fit_predict(distance_matrix)

# --- 3. (Optional) Visualise dendrogram -------------------------------
# from scipy.cluster.hierarchy import dendrogram, linkage
# linked = linkage(distance_matrix, 'average')
# dendrogram(linked, labels=[item['videoId'] for item in data])
# plt.show()

# --- 4. Group results -------------------------------------------------
clusters = defaultdict(list)
for idx, label in enumerate(labels):
    clusters[label].append({"videoId": data[idx]["videoId"], "text": data[idx]["text"]})

# --- 5. Display -------------------------------------------------------
for cid, items in clusters.items():
    print(f"\n🔷 Cluster {cid} (size {len(items)})")
    for it in items:
        print(f"   - {it['videoId']}: {it['text'][:60]}...")
