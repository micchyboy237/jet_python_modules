from transformers import T5Tokenizer, T5Model
import torch
import faiss
import numpy as np
from typing import List, TypedDict


class SearchResult(TypedDict):
    rank: int
    score: float
    text: str


class KeywordVectorSearch:
    def __init__(self, model_name: str = "t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name)
        self.index = None
        self.words: List[str] = []

    def encode_words(self, words: List[str]) -> np.ndarray:
        """Encode a list of words into embeddings."""
        inputs = self.tokenizer(
            words, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.encoder(**inputs).last_hidden_state
            embeddings = outputs.mean(dim=1).numpy()
        return embeddings

    def build_index(self, words: List[str]):
        """Build FAISS index for vector search."""
        self.words = words
        embeddings = self.encode_words(words)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        return embeddings

    def search(self, query: str, k: int = 5) -> List[SearchResult]:
        """Search for the k closest words to the query."""
        if not self.index or not self.words:
            raise ValueError(
                "Index or word list not initialized. Call build_index first.")
        query_embedding = self.encode_words([query])
        distances, indices = self.index.search(query_embedding, k)
        results = [
            SearchResult(rank=i + 1, score=float(d), text=self.words[idx])
            for i, (idx, d) in enumerate(zip(indices[0], distances[0]))
        ]
        # Lower distance is better
        return sorted(results, key=lambda x: x["score"], reverse=False)
