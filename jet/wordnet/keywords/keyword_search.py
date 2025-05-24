# jet/wordnet/keywords/roberta_keyword_search.py

from typing import List, Dict
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Load a SentenceTransformer model fine-tuned for semantic similarity
model = SentenceTransformer("all-mpnet-base-v2")

# Use MPS if available (for Mac M1)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)


def get_embedding(text: str) -> torch.Tensor:
    """Generate embedding for a given text."""
    return model.encode(text, convert_to_tensor=True, device=device)


def search_keywords(topic: str, texts: List[str], top_k: int = 5) -> List[Dict]:
    """Rank texts based on semantic similarity to the topic."""
    topic_embedding = get_embedding(topic)
    text_embeddings = model.encode(
        texts, convert_to_tensor=True, device=device)

    similarities = util.pytorch_cos_sim(topic_embedding, text_embeddings)[0]
    top_k = min(top_k, len(texts))
    top_scores, top_indices = torch.topk(similarities, k=top_k)

    results = [
        {"text": texts[i], "score": top_scores[idx].item()}
        for idx, i in enumerate(top_indices)
    ]
    return results
