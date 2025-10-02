import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from jet.libs.llama_cpp.embeddings import LlamacppEmbedding
from jet.llm.models import OLLAMA_MODEL_NAMES

class VectorSearch:
    """A vector search engine using sentence transformers."""
    def __init__(self, model: str | OLLAMA_MODEL_NAMES = "all-minilm:33m", truncate_dim: Optional[int] = None, batch_size=32):
        self.documents: List[str] = []
        self.vectors: np.ndarray = None

        client = LlamacppEmbedding(model=model)
        self.embedding_function = client.get_embedding_function(return_format="numpy", batch_size=batch_size, show_progress=True)

    def add_documents(self, documents: List[str]) -> None:
        """Add documents and their vector representations."""
        self.documents = documents
        self.vectors = self.embedding_function(documents)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for documents most similar to the query."""
        if not self.documents:
            return []
        query_vector = self.embedding_function(query).reshape(1, -1)
        similarities = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.documents[i], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    model: OLLAMA_MODEL_NAMES = "embeddinggemma"
    search_engine = VectorSearch(model)
    sample_docs = [
        "Fresh organic apples from local farms",
        "Handpicked strawberries sweet and juicy",
        "Premium quality oranges rich in vitamin C",
        "Crisp lettuce perfect for salads",
        "Organic bananas ripe and ready to eat"
    ]
    search_engine.add_documents(sample_docs)
    queries = [
        "organic fruit",
        "sweet strawberries",
        "fresh salad ingredients"
    ]
    for query in queries:
        results = search_engine.search(query)
        print(f"\nQuery: {query}")
        print("Top matches:")
        for doc, score in results:
            print(f"- {doc} (score: {score:.3f})")
