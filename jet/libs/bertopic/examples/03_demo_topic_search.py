from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()
from jet.adapters.bertopic.utils import extract_topics_with_query
from jet.libs.bertopic.examples.doc_samples import DOCS_LG
from jet.logger import logger


def run_query_search_example():
    logger.info(
        "Starting Guided/Query BERTopic search example utilizing Jet Adapters..."
    )

    documents = DOCS_LG

    search_query = "renewable energy and healtcare"
    logger.info(f"Using search query: '{search_query}'")

    try:
        # Reuses full batched llama.cpp embeddings under the hood
        topics, query_result = extract_topics_with_query(
            docs=documents, query=search_query, top_k=5, min_topic_size=2, verbose=True
        )

        logger.info("Search results mapped successfully:")
        for rank, (topic_id, score) in enumerate(
            zip(query_result["topic_ids"], query_result["probabilities"])
        ):
            logger.info(
                f"Rank {rank + 1} -> Topic ID: {topic_id} | Relevance Score: {score:.4f}"
            )

    except Exception as e:
        logger.error(
            f"An error occurred during query topic extraction: {e}", exc_info=True
        )


if __name__ == "__main__":
    run_query_search_example()
