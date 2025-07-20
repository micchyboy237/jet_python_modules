from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/allenai-specter')
embeddings = model.encode(sentences)
print(embeddings)