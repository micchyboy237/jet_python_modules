from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    # Use a persistent database name
    demo_db_name = "jet_vector_persistent"

    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")
    # Set overwrite_db=False to keep data between runs
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=False)

    try:
        table_name = "demo_items"
        dimension = 768

        # Check if table exists, if not create it and populate it
        if table_name not in client.get_all_tables():
            logger.info(
                f"Creating vector table '{table_name}' with dimension {dimension}..."
            )
            client.create_vector_table(
                table_name=table_name,
                dimension=dimension,
                additional_columns={"content": "TEXT"},
            )

            sample_texts = [
                "The dog is barking loudly at the mailman.",
                "The cat is purring softly on the windowsill.",
                "Python is a versatile programming language for data science.",
                "PostgreSQL is a powerful open-source relational database.",
                "The bear is growling in the forest during winter.",
            ]

            logger.info("Generating and saving real embeddings (One-time cost)...")
            embeddings = embed(sample_texts, prefix=EMBED_DOC_PREFIX)

            sample_data = [
                {"id": str(i + 1), "content": text, "embedding": emb.tolist()}
                for i, (text, emb) in enumerate(zip(sample_texts, embeddings))
            ]

            client.create_rows(table_name, sample_data)
            logger.success(f"Inserted {len(sample_data)} items.")

            logger.info("Creating HNSW index for Cosine similarity...")
            client.create_index(table_name, metric="cosine", method="hnsw")

            logger.info("Creating Full-Text Search index...")
            client.create_fts_index(table_name, column_name="content")
        else:
            logger.info(
                f"Table '{table_name}' already exists. Reusing existing embeddings."
            )
            count = client.count_embeddings(table_name)
            logger.success(f"Found {count} existing embeddings in database.")

        # Perform multiple searches (Only the query is embedded now!)
        queries = [
            "A canine is making noise",
            "Coding in Python",
            "Wildlife in the woods",
        ]

        for query_text in queries:
            logger.info(f"\nSearching for: '{query_text}'")

            # Only embed the query
            query_embedding = embed(query_text, prefix=EMBED_QUERY_PREFIX).tolist()

            results = client.search(
                table_name=table_name,
                query_embedding=query_embedding,
                top_k=2,
                metric="cosine",
            )

            print(f"  Top Results for '{query_text}':")
            for res in results:
                print(
                    f"    - ID: {res['id']} | Score: {res['score']:.4f} | {res['content'][:50]}..."
                )

    except Exception as e:
        logger.error(f"An error occurred during demo: {e}")
    finally:
        # We do NOT delete the DB here so we can run it again and see the reuse
        logger.info("\nDemo finished. Database persists for next run.")
        client.close()


if __name__ == "__main__":
    main()
