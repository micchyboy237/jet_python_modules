"""
BERTopic Usage Example: Encode Documents to Embeddings

This example demonstrates how to encode a list of documents into dense vector 
embeddings using a pre-initialized SentenceTransformer model. This is useful 
for pre-computing embeddings or inspecting the embedding space.

Key Features:
- Standalone function for document encoding
- Support for batch processing
- Embedding inspection and analysis
- Integration with similarity calculations

Usage:
    python 02_encode_documents.py
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def encode_documents(embedding_model: SentenceTransformer, documents: List[str]) -> np.ndarray:
    """
    Encodes a list of documents into dense vector embeddings.
    
    Args:
        embedding_model: Pre-initialized SentenceTransformer model.
        documents: List of input text strings.
    
    Returns:
        NumPy array of embeddings (shape: [n_docs, embedding_dim]).
    
    Example:
        emb_model = create_embedding_model()
        embeddings = encode_documents(emb_model, SAMPLE_DOCS[:5])
        print(embeddings.shape)  # e.g., (5, 384)
    """
    return embedding_model.encode(documents)


def analyze_embeddings(embeddings: np.ndarray, documents: List[str]) -> None:
    """
    Analyzes and displays information about the generated embeddings.
    
    Args:
        embeddings: Array of document embeddings.
        documents: List of original documents.
    """
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Number of documents: {len(documents)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Calculate pairwise similarities
    similarities = cosine_similarity(embeddings)
    
    print(f"\nPairwise similarities:")
    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            sim = similarities[i, j]
            print(f"  Doc {i} vs Doc {j}: {sim:.4f}")
            print(f"    '{documents[i][:50]}...'")
            print(f"    '{documents[j][:50]}...'")


def main():
    """Demonstrate document encoding and analysis."""
    print("=== BERTopic Document Encoding Example ===\n")
    
    # Sample documents for testing
    sample_docs = [
        "Apple stock rises amid market volatility.",
        "Berkshire Hathaway invests in tech startups.",
        "Nasdaq index hits new high on earnings reports.",
        "Tesla announces new battery technology.",
        "Inflation concerns impact bond yields.",
        "Crypto market rebounds after regulatory news.",
        "Amazon expands cloud services in Europe.",
        "Gold prices surge due to geopolitical tensions."
    ]
    
    # Create embedding model
    print("1. Creating embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"   Model: {embedding_model}")
    
    # Encode documents
    print("\n2. Encoding documents...")
    embeddings = encode_documents(embedding_model, sample_docs)
    
    # Analyze embeddings
    print("\n3. Analyzing embeddings...")
    analyze_embeddings(embeddings, sample_docs)
    
    # Show embedding statistics
    print(f"\n4. Embedding statistics:")
    print(f"   Mean embedding value: {np.mean(embeddings):.4f}")
    print(f"   Std embedding value: {np.std(embeddings):.4f}")
    print(f"   Min embedding value: {np.min(embeddings):.4f}")
    print(f"   Max embedding value: {np.max(embeddings):.4f}")
    
    # Demonstrate batch processing
    print(f"\n5. Batch processing example:")
    batch_size = 3
    for i in range(0, len(sample_docs), batch_size):
        batch_docs = sample_docs[i:i + batch_size]
        batch_embeddings = encode_documents(embedding_model, batch_docs)
        print(f"   Batch {i//batch_size + 1}: {len(batch_docs)} docs, shape {batch_embeddings.shape}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
