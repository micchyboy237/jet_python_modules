import logging
from sklearn.metrics.pairwise import cosine_similarity
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

def example_extract_embeddings():
    """Demonstrate extracting embeddings from documents."""
    logging.info("Starting extract embeddings example...")
    topic_model = BERTopic()
    
    # Extract single document embedding
    single_doc = "a document"
    logging.info(f"Extracting embedding for single document: {single_doc}")
    single_embedding = topic_model._extract_embeddings(single_doc)
    logging.info(f"Single embedding shape: {single_embedding.shape}")
    
    # Extract multiple document embeddings
    multiple_docs = ["something different", "another document"]
    logging.info(f"Extracting embeddings for multiple documents: {multiple_docs}")
    multiple_embeddings = topic_model._extract_embeddings(multiple_docs)
    logging.info(f"Multiple embeddings shape: {multiple_embeddings.shape}")
    
    # Compute cosine similarity
    sim_matrix = cosine_similarity(single_embedding, multiple_embeddings)[0]
    logging.info(f"Cosine similarities: {sim_matrix}")
    
    return single_embedding, multiple_embeddings

def example_extract_embeddings_compare():
    """Demonstrate comparing BERTopic embeddings with sentence transformer embeddings."""
    logging.info("Starting extract embeddings comparison example...")
    topic_model = BERTopic()
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = ["some document"]
    
    logging.info("Extracting BERTopic embeddings...")
    bertopic_embeddings = topic_model._extract_embeddings(docs)
    logging.info(f"BERTopic embeddings shape: {bertopic_embeddings.shape}")
    
    logging.info("Extracting sentence transformer embeddings...")
    sentence_embeddings = embedding_model.encode(docs, show_progress_bar=False)
    logging.info(f"Sentence transformer embeddings shape: {sentence_embeddings.shape}")
    
    return bertopic_embeddings, sentence_embeddings

def example_extract_incorrect_embeddings():
    """Demonstrate handling incorrect embeddings extraction."""
    logging.info("Starting incorrect embeddings extraction example...")
    try:
        model = BERTopic(language="Unknown language")
        model.fit(["some document"])
        logging.error("Expected ValueError not raised for unknown language")
    except ValueError as e:
        logging.info(f"Correctly caught ValueError: {str(e)}")
    
    return None

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_extract_embeddings()
    example_extract_embeddings_compare()
    example_extract_incorrect_embeddings()
    logging.info("Embeddings usage examples completed successfully.")
