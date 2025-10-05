"""
BERTopic Usage Example: Create Embedding Model

This example demonstrates how to create and configure a SentenceTransformer 
embedding model for BERTopic. The embedding model converts text documents 
into dense vector representations that capture semantic meaning.

Key Features:
- Load pre-trained SentenceTransformer models
- Support for different model sizes and capabilities
- Easy model switching for different use cases
- Type hints for better code clarity

Usage:
    python 01_embedding_model.py
"""

from typing import List
from sentence_transformers import SentenceTransformer


def create_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Creates a SentenceTransformer embedding model.
    
    Args:
        model_name: Name of the pre-trained model (e.g., "all-MiniLM-L6-v2" for lightweight, 
                   or "BAAI/bge-base-en-v1.5" for better semantics).
    
    Returns:
        Initialized SentenceTransformer instance.
    
    Example:
        emb_model = create_embedding_model("all-MiniLM-L6-v2")
        embeddings = emb_model.encode(["Sample doc 1", "Sample doc 2"])
    """
    return SentenceTransformer(model_name)


def main():
    """Demonstrate embedding model creation and usage."""
    print("=== BERTopic Embedding Model Example ===\n")
    
    # Sample documents for testing
    sample_docs = [
        "Apple stock rises amid market volatility.",
        "Berkshire Hathaway invests in tech startups.",
        "Nasdaq index hits new high on earnings reports.",
        "Tesla announces new battery technology.",
        "Inflation concerns impact bond yields."
    ]
    
    # Create different embedding models
    print("1. Creating lightweight embedding model...")
    light_model = create_embedding_model("all-MiniLM-L6-v2")
    print(f"   Model: {light_model}")
    print(f"   Embedding dimension: {light_model.get_sentence_embedding_dimension()}")
    
    # Encode sample documents
    print("\n2. Encoding sample documents...")
    embeddings = light_model.encode(sample_docs)
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Number of documents: {len(sample_docs)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    
    # Show similarity between first two documents
    from sklearn.metrics.pairwise import cosine_similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    print(f"\n3. Similarity between first two documents: {similarity:.4f}")
    
    # Demonstrate with a more powerful model (commented out to avoid download)
    print("\n4. For better semantic understanding, you can use:")
    print("   model = create_embedding_model('BAAI/bge-base-en-v1.5')")
    print("   # This model provides better semantic understanding but is larger")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
