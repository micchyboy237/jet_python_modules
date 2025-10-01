from typing import List, Union, Dict
from ollama import embed
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingSearch:
    """A class to handle text embedding and document search using cosine similarity."""
    
    def __init__(self, model: str = "llama3.2"):
        """Initialize with a specified embedding model."""
        self.model = model

    def get_embedding(self, text: Union[str, List[str]]) -> List[float]:
        """
        Generate embedding for input text or list of texts.
        
        Args:
            text: A single string or list of strings to embed.
            
        Returns:
            List of embeddings for the input text(s).
        """
        response = embed(model=self.model, input=text)
        return response["embeddings"]

    def search_documents(self, query: str, documents: List[str], top_k: int = 3) -> List[Dict]:
        """
        Search for documents most similar to the query based on embeddings.
        
        Args:
            query: The query string to search for.
            documents: List of document strings to search through.
            top_k: Number of top similar documents to return.
            
        Returns:
            List of dictionaries with document text and similarity score, sorted by score.
        """
        # Combine query and documents for a single embedding call
        all_texts = [query] + documents
        all_embeddings = self.get_embedding(all_texts)
        
        # Split embeddings
        query_embedding = all_embeddings[0]
        doc_embeddings = all_embeddings[1:]
        
        # Calculate similarities using sklearn
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        results = [
            {"document": doc, "score": float(score)}
            for doc, score in zip(documents, similarities)
        ]
        
        # Sort by score in descending order and return top_k
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# Sample usage
if __name__ == "__main__":
    # Initialize the search class
    searcher = EmbeddingSearch(model="embeddinggemma")
    
    # Sample query and documents
    query = "Which planet is known as the Red Planet?"
    documents = [
        "Venus is often called Earth's twin because of its similar size and proximity.",
        "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
        "Jupiter, the largest planet in our solar system, has a prominent red spot.",
        "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
    ]
    
    # Perform document search
    results = searcher.search_documents(query, documents, top_k=4)
    
    # Print results
    print("Query:", query)
    for result in results:
        print(f"Document: {result['document']}, Score: {result['score']:.4f}")