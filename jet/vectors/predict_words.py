from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

model = SentenceTransformer("all-MiniLM-L12-v2")

# Sample vocabulary
words = ["dog", "cat", "house", "tree", "running", "jumps", "sleeping"]

# Compute embeddings for words
word_embeddings = model.encode(words)
word_embeddings = np.array(word_embeddings).astype("float32")

# Build FAISS index
index = faiss.IndexFlatL2(word_embeddings.shape[1])
index.add(word_embeddings)

# Query sentence fragment
query_embedding = model.encode(["The animal is"])[0].reshape(1, -1)
_, indices = index.search(query_embedding.astype("float32"), 1)

predicted_word = words[indices[0][0]]
print(predicted_word)  # Might return a related word like "cat" or "dog"
