import os
from typing import List
from jet.examples.resume_helpers.base import VectorSearch, load_resume_markdown, SearchResult
from jet.file.utils import save_file
from jet.logger import logger

output_dir = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
resume_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/examples/resume_helpers/data/sample_resume.md"


def main():
    """Main function to load resume and run vector search."""
    try:
        # Initialize VectorSearch
        vector_search = VectorSearch()

        # Load resume markdown
        resume = load_resume_markdown(resume_path)
        documents = [resume]

        # Preprocess and index
        vector_search.preprocess_and_index(documents, chunk_size=500)
        logger.info("Resume indexed successfully")

        # Example job interviewer query
        query = "Does candidate C123 have sufficient React experience for a Senior Frontend Developer role?"
        results: List[SearchResult] = vector_search.search(
            query, top_k=3, hybrid_weight=0.5)

        # Display results
        logger.info(f"Query: {query}")
        for i, result in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"Chunk ID: {result['chunk']['id']}")
            logger.info(f"Score: {result['score']:.4f}")
            logger.info(f"Text: {result['chunk']['text'][:200]}...")
            logger.info(f"Metadata: {result['chunk']['metadata']}")
            logger.info("-" * 50)

        # Evaluate retrieval
        relevant_chunk_ids = [f"C123_{i}" for i in range(
            len(vector_search.chunks)) if "React" in vector_search.chunks[i]["text"]]
        ndcg = vector_search.evaluate_retrieval(
            query, relevant_chunk_ids, top_k=3)
        logger.info(f"NDCG Score: {ndcg:.4f}")

    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
