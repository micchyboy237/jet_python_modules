from jet.llm.mlx.models import get_embedding_size
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
from scipy.spatial.distance import hamming

# 1. Specify embed model and get dimensions
embed_model = "mixedbread-ai/mxbai-embed-large-v1"
dimensions = get_embedding_size(embed_model) // 2

# 2. Load model
model = SentenceTransformer(embed_model, truncate_dim=dimensions)

# Query and documents
query = "A man is eating a piece of bread"
docs = [
    "A man is eating food.",
    "A man is eating pasta.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
]

# 3. Encode
query_embedding = model.encode(
    query, prompt_name="query", convert_to_numpy=True)
docs_embeddings = model.encode(docs, convert_to_numpy=True)

# 4. Debug: Print shapes
print("Query embedding shape:", query_embedding.shape)
print("Docs embeddings shape:", docs_embeddings.shape)

# 5. Ensure correct shape for quantization (2D array)
if query_embedding.ndim == 1:
    query_embedding = query_embedding.reshape(1, -1)
if docs_embeddings.ndim == 1:
    docs_embeddings = docs_embeddings.reshape(-1, dimensions)

# 6. Quantize the embeddings
binary_query_embedding = quantize_embeddings(
    query_embedding, precision="ubinary")
binary_docs_embeddings = quantize_embeddings(
    docs_embeddings, precision="ubinary")

# 7. Debug: Print shapes after quantization
print("Binary query embedding shape:", binary_query_embedding.shape)
print("Binary docs embeddings shape:", binary_docs_embeddings.shape)

# 8. Compute Hamming distances for binary embeddings
# Convert binary embeddings to boolean arrays for Hamming distance
binary_query_embedding = binary_query_embedding > 0
binary_docs_embeddings = binary_docs_embeddings > 0

# Compute Hamming distance for each document
hamming_distances = np.array(
    [hamming(binary_query_embedding[0], doc_emb) for doc_emb in binary_docs_embeddings])

# Convert to similarity
binary_similarities = 1 - hamming_distances

# 9. Print results
print('Binary similarities:', binary_similarities)

# Optional: Compare with floating-point similarities
float_similarities = cos_sim(query_embedding, docs_embeddings)
print('Float similarities:', float_similarities)
