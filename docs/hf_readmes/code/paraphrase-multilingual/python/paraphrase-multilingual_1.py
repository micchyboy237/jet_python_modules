from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L12-v2')
embeddings = model.encode(sentences)
print(embeddings)