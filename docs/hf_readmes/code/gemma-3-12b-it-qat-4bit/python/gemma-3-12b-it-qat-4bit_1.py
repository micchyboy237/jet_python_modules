from mlx_embeddings import load, generate
import mlx.core as mx

model, tokenizer = load("mlx-community/all-MiniLM-L6-v2-6bit")

# For text embeddings
output = generate(model, processor, texts=["I like grapes", "I like fruits"])
embeddings = output.text_embeds  # Normalized embeddings

# Compute dot product between normalized embeddings
similarity_matrix = mx.matmul(embeddings, embeddings.T)

print("Similarity matrix between texts:")
print(similarity_matrix)