from transformers import GPT2Model, GPT2Tokenizer
from sklearn.cluster import KMeans
import torch
import numpy as np


def create_context_windows(tokens, window_size=5):
    context_windows = []
    for i in range(len(tokens) - window_size + 1):
        window = tokens[i:i + window_size]
        context_windows.append(window)
    return context_windows


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')


def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state


def cluster_embeddings(texts, num_clusters=5):
    embeddings = [get_embeddings(text).detach().numpy() for text in texts]
    embeddings = np.concatenate(embeddings)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(embeddings)
    return kmeans.labels_


def predict_masked_token(text, masked_index):
    inputs = tokenizer.encode(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs)
    predictions = outputs[0]
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.decode([predicted_index])
    return predicted_token
