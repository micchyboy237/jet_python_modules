from jet.adapters.keybert import KeyBERT
from jet.adapters.llama_cpp.embed_utils import embed
from sklearn.cluster import KMeans

kw_model = KeyBERT()

# Imagine these are keywords extracted from 1000 documents
extracted_keywords = [
    "machine learning",
    "AI",
    "deep learning",
    "neural networks",
    "football",
    "soccer",
    "NBA",
    "basketball",
    "stock market",
    "investing",
    "crypto",
    "bitcoin",
]

# Get embeddings for these keywords using jet's llama.cpp embedder
embeddings = embed(extracted_keywords)

# Cluster them into 3 topics
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Print clusters
for i in range(3):
    cluster_words = [
        extracted_keywords[j]
        for j in range(len(extracted_keywords))
        if clusters[j] == i
    ]
    print(f"Topic {i + 1}: {cluster_words}")
