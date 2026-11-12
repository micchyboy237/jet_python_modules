# jet_python_modules/jet/db/postgres/examples/demo_multi_query_search.py
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    demo_db_name = "jet_vector_multi_demo"
    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=True)

    try:
        table_name = "tech_items"
        dimension = 768

        client.create_vector_table(
            table_name, dimension, additional_columns={"content": "TEXT"}
        )

        # Data with specific terms
        texts = [
            "Python is great for data science.",
            "Java is used for enterprise backends.",
            "JavaScript powers the modern web.",
            "C++ is ideal for game engines.",
            "Rust offers memory safety without garbage collection.",
        ]

        from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX
        from jet.adapters.llama_cpp.embed_utils import embed

        embeddings = embed(texts, prefix=EMBED_DOC_PREFIX)
        data = [
            {"id": str(i), "content": t, "embedding": e.tolist()}
            for i, (t, e) in enumerate(zip(texts, embeddings))
        ]

        client.bulk_ingest(table_name, data, dimension=dimension)
        client.create_index(table_name, metric="cosine", method="hnsw")

        # Scenario: User searches for "Web coding"
        # We expand this to include synonyms to catch more relevant results
        query_expansions = ["Web coding", "Frontend development", "Browser scripting"]

        logger.info(f"Performing multi-query search for: {query_expansions}")

        results = client.multi_query_search(
            table_name=table_name,
            query_texts=query_expansions,
            top_k=3,
            num_candidates=5,
        )

        print("\n--- Multi-Query Search Results ---")
        for res in results:
            print(
                f"ID: {res['id']} | RRF Score: {res['rrf_score']:.4f} | {res['content']}"
            )

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client.delete_db(confirm=True)
        client.close()


if __name__ == "__main__":
    main()
