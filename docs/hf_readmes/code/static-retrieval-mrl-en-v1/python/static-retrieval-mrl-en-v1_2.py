from sentence_transformers import SentenceTransformer

model = SentenceTransformer("tomaarsen/static-retrieval-mrl-en-v1", truncate_dim=256)
embeddings = model.encode([
    "what is the difference between chronological order and spatial order?",
    "can lavender grow indoors?"
])
print(embeddings.shape)
# => (2, 256)