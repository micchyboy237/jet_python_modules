# jet_python_modules/jet/db/postgres/examples/demo_bulk_ingestion_jet.py
import time

from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    demo_db_name = "jet_vector_bulk_demo"
    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")

    # Overwrite to start fresh
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=True)

    try:
        table_name = "bulk_items"
        dimension = 768

        logger.info(f"Creating vector table '{table_name}'...")
        client.create_vector_table(
            table_name=table_name,
            dimension=dimension,
            additional_columns={"content": "TEXT", "source": "TEXT"},
        )

        # Simulate 1,000 documents
        num_docs = 1000
        logger.info(f"Generating {num_docs} sample documents and embeddings...")

        # For demo speed, we'll generate embeddings in batches
        sample_texts = [
            f"Document number {i} containing unique semantic content for testing."
            for i in range(num_docs)
        ]

        start_embed = time.time()
        # Embed in large batches for efficiency
        embeddings = embed(sample_texts, prefix=EMBED_DOC_PREFIX, batch_size=128)
        embed_time = time.time() - start_embed
        logger.success(f"Embedding generation took {embed_time:.2f}s")

        # Prepare data for ingestion
        data_to_ingest = [
            {
                "id": f"doc_{i}",
                "content": text,
                "source": "demo_generator",
                "embedding": emb.tolist(),
            }
            for i, (text, emb) in enumerate(zip(sample_texts, embeddings))
        ]

        logger.info(f"Starting bulk ingestion of {num_docs} records...")
        start_ingest = time.time()

        count = client.bulk_ingest(table_name, data_to_ingest, dimension=dimension)

        ingest_time = time.time() - start_ingest
        logger.success(
            f"Bulk ingestion completed: {count} records in {ingest_time:.2f}s"
        )
        logger.info(f"Throughput: {count / ingest_time:.0f} records/second")

        # Verify count
        db_count = client.count_embeddings(table_name)
        logger.info(f"Verified record count in DB: {db_count}")

        # Test a search to ensure index works (if created)
        logger.info("Creating HNSW index for performance...")
        client.create_index(table_name, metric="cosine", method="hnsw")

        query_emb = embed("Document number 500", prefix=EMBED_DOC_PREFIX).tolist()
        res = client.search(table_name, query_emb, top_k=1)
        logger.info(f"Search test result: {res[0]['id']}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("Cleaning up...")
        client.delete_db(confirm=True)
        client.close()


if __name__ == "__main__":
    main()
