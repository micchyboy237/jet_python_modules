from jet.adapters.bertopic import BERTopic
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Example 1: Extracting embeddings for a single document
def example_single_document_embedding():
    # Given: A pre-trained BERTopic model and a single document
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    document = "The quick brown fox jumps over the lazy dog"
    
    # When: Extracting embeddings for the single document
    single_embedding = topic_model._extract_embeddings(document)
    
    # Then: Verify the embedding shape and value range
    print(f"Single document embedding shape: {single_embedding.shape}")
    print(f"Min value: {np.min(single_embedding):.2f}, Max value: {np.max(single_embedding):.2f}")

# Example 2: Extracting embeddings for multiple documents and computing similarity
def example_multiple_documents_embedding():
    # Given: A pre-trained BERTopic model and multiple documents
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    documents = [
        "A fast fox runs through the forest",
        "The dog sleeps peacefully under the tree"
    ]
    reference_doc = "A quick fox jumps over obstacles"
    
    # When: Extracting embeddings for multiple documents and computing cosine similarity
    multiple_embeddings = topic_model._extract_embeddings(documents)
    reference_embedding = topic_model._extract_embeddings(reference_doc)
    sim_matrix = cosine_similarity(reference_embedding, multiple_embeddings)[0]
    
    # Then: Verify the embeddings shape and similarity scores
    print(f"Multiple documents embedding shape: {multiple_embeddings.shape}")
    print(f"Similarity with first document: {sim_matrix[0]:.2f}")
    print(f"Similarity with second document: {sim_matrix[1]:.2f}")

# Example 3: Comparing BERTopic embeddings with SentenceTransformer embeddings
def example_compare_embeddings():
    # Given: A pre-trained BERTopic model and a SentenceTransformer model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    topic_model = BERTopic(embedding_model=embedding_model)
    document = ["The sun sets slowly behind the mountain"]
    
    # When: Extracting embeddings using both models
    bertopic_embeddings = topic_model._extract_embeddings(document)
    sentence_embeddings = embedding_model.encode(document, show_progress_bar=False)
    
    # Then: Verify the embeddings are equivalent
    print(f"BERTopic embedding shape: {bertopic_embeddings.shape}")
    print(f"SentenceTransformer embedding shape: {sentence_embeddings.shape}")
    print(f"Embeddings are equal: {np.array_equal(bertopic_embeddings, sentence_embeddings)}")

if __name__ == "__main__":
    print("Running Example 1: Single document embedding")
    example_single_document_embedding()
    print("\nRunning Example 2: Multiple documents embedding")
    example_multiple_documents_embedding()
    print("\nRunning Example 3: Compare embeddings")
    example_compare_embeddings()
