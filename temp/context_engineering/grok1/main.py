from rag import rag_generation
from config import EMBEDDING_MODEL, LLM_MODEL

# Example usage
documents = [
    "Document 1: The capital of France is Paris.",
    "Document 2: The Eiffel Tower is in Paris.",
    "Document 3: France is in Europe."
]
query = "What is the capital of France?"

# Call the RAG function
answer = rag_generation(query, documents)
print("Generated Answer:", answer)