from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
embeddings = model.encode(sentences)
print(embeddings)