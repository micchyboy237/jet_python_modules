from jet.libs.bertopic.monkey_patches.add_check_array import init_patch

init_patch()


from jet.adapters.bertopic.utils import extract_topics_without_query
from jet.libs.bertopic.examples.doc_samples import DOCS_LG
from jet.logger import logger


def run_basic_example():
    logger.info("Starting Basic BERTopic extraction example utilizing Jet Adapters...")

    # Sample documents dataset
    documents = DOCS_LG

    try:
        # Reuses the optimized pipeline with embedded llama.cpp configurations
        topics, topic_info = extract_topics_without_query(
            docs=documents, min_topic_size=2, verbose=True
        )

        logger.info("Topic extraction complete! Results summary:")
        for topic_id, words in topic_info.items():
            logger.info(f"Topic ID {topic_id}: {words[:5]}")

    except Exception as e:
        logger.error(
            f"An error occurred during basic topic extraction: {e}", exc_info=True
        )


if __name__ == "__main__":
    run_basic_example()
