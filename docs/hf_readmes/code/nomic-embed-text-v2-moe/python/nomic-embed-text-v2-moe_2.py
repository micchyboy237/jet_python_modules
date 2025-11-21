from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
sentences = ["Hello!", "¡Hola!"]
embeddings = model.encode(sentences, prompt_name="passage")
print(embeddings.shape)
# (2, 768)

similarity = model.similarity(embeddings[0], embeddings[1])
print(similarity)
# tensor([[0.9118]])