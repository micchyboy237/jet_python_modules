import logging
import numpy as np
from scipy.sparse import csr_matrix
from bertopic._utils import (
    check_documents_type,
    check_embeddings_shape,
    MyLogger,
    select_topic_representation,
    get_unique_distances,
)

def example_logger_configuration():
    """Demonstrate configuring the MyLogger utility."""
    logger = MyLogger()
    logger.configure("DEBUG")
    logging.info("Logger configured to DEBUG level")
    logger.logger.debug("This is a debug message")
    logger.configure("WARNING")
    logging.info("Logger reconfigured to WARNING level")
    logger.logger.warning("This is a warning message")
    return logger

def example_check_documents_type():
    """Demonstrate checking document types with check_documents_type utility."""
    valid_docs = ["doc1", "doc2", "doc3"]
    invalid_docs = ["single_doc", [None], 42]
    
    logging.info("Checking valid document list...")
    try:
        check_documents_type(valid_docs)
        logging.info("Valid documents passed type check")
    except TypeError:
        logging.error("Valid documents failed type check")
    
    for doc in invalid_docs:
        logging.info(f"Checking invalid input: {doc}")
        try:
            check_documents_type(doc)
            logging.error(f"Invalid input {doc} unexpectedly passed")
        except TypeError:
            logging.info(f"Invalid input {doc} correctly raised TypeError")
    
    return valid_docs

def example_check_embeddings_shape():
    """Demonstrate checking embeddings shape compatibility with documents."""
    documents = ["doc1", "doc2"]
    embeddings = np.array([[1, 2, 3], [4, 5, 6]])
    
    logging.info("Checking embeddings shape...")
    try:
        check_embeddings_shape(embeddings, documents)
        logging.info("Embeddings shape check passed")
    except ValueError:
        logging.error("Embeddings shape check failed")
    
    invalid_embeddings = np.array([[1, 2, 3]])
    logging.info("Checking invalid embeddings shape...")
    try:
        check_embeddings_shape(invalid_embeddings, documents)
        logging.error("Invalid embeddings shape unexpectedly passed")
    except ValueError:
        logging.info("Invalid embeddings shape correctly raised ValueError")
    
    return embeddings

def example_get_unique_distances():
    """Demonstrate creating unique distances with get_unique_distances utility."""
    distances = [0, 0, 0.5, 0.75, 1, 1]
    noise_max = 1e-7
    
    logging.info(f"Processing distances: {distances}")
    unique_dists = get_unique_distances(np.array(distances, dtype=float), noise_max=noise_max)
    logging.info(f"Unique distances: {unique_dists}")
    
    assert len(unique_dists) == len(distances)
    assert len(np.unique(unique_dists)) == len(distances)
    
    return unique_dists

def example_select_topic_representation():
    """Demonstrate selecting topic representation with select_topic_representation utility."""
    ctfidf_embeddings = np.array([[1, 1, 1]])
    ctfidf_embeddings_sparse = csr_matrix(
        (ctfidf_embeddings.reshape(-1).tolist(), ([0, 0, 0], [0, 1, 2])),
        shape=ctfidf_embeddings.shape,
    )
    topic_embeddings = np.array([[2, 2, 2]])
    
    logging.info("Selecting topic representation with topic_embeddings and use_ctfidf=False...")
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, topic_embeddings, use_ctfidf=False)
    logging.info(f"Selected representation: {repr_}, ctfidf_used: {ctfidf_used}")
    
    logging.info("Selecting topic representation with ctfidf_embeddings and use_ctfidf=True...")
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings, None, use_ctfidf=True)
    logging.info(f"Selected representation: {repr_}, ctfidf_used: {ctfidf_used}")
    
    logging.info("Selecting sparse ctfidf representation with output_ndarray=False...")
    repr_, ctfidf_used = select_topic_representation(ctfidf_embeddings_sparse, None, use_ctfidf=True, output_ndarray=False)
    logging.info(f"Selected representation type: {type(repr_)}, ctfidf_used: {ctfidf_used}")
    
    return repr_, ctfidf_used

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_logger_configuration()
    example_check_documents_type()
    example_check_embeddings_shape()
    example_get_unique_distances()
    example_select_topic_representation()
    logging.info("All utils usage examples completed successfully.")
