from jet.adapters.keybert import KeyBERT

# Initialize KeyBERT
kw_model = KeyBERT()

docs = [
    "Deep learning is a subset of machine learning.",
    "Natural language processing helps computers understand text.",
]

# 1. Compute embeddings once using jet's optimized llama.cpp embedder
# Note: KeyBERT expects document embeddings and word embeddings.
# For simplicity in this demo, we use the standard extract_keywords which handles internal embedding,
# but if you wanted to use pre-computed ones, you'd use kw_model.extract_embeddings()
# which internally uses the configured embedder.

# Let's demonstrate the speed of batch processing vs single
import time

start = time.time()
for doc in docs:
    kw_model.extract_keywords(doc)
single_time = time.time() - start

start = time.time()
kw_model.extract_keywords(docs)
batch_time = time.time() - start

print(f"Single doc processing time: {single_time:.4f}s")
print(f"Batch processing time: {batch_time:.4f}s")
print(f"Speedup: {single_time / batch_time:.2f}x")
