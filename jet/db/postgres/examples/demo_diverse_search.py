# jet_python_modules/jet/db/postgres/examples/demo_diverse_search.py
from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    demo_db_name = "jet_vector_diversity_demo_v2"

    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=True)

    try:
        table_name = "diverse_items"
        dimension = 768

        logger.info(f"Creating vector table '{table_name}'...")
        client.create_vector_table(
            table_name=table_name,
            dimension=dimension,
            additional_columns={"content": "TEXT"},
        )

        # More data with heavy clumping around "Python programming"
        sample_texts = [
            "Python is a great programming language.",  # 0
            "I love coding in Python.",  # 1
            "Python scripts are easy to write.",  # 2
            "Learning Python is fun.",  # 3
            "Java is also a popular language.",  # 4
            "C++ is used for high-performance games.",  # 5
            "JavaScript runs in the browser.",  # 6
            "SQL is for database queries.",  # 7
            "Rust is memory-safe.",  # 8
            "Go is great for concurrency.",  # 9
        ]

        logger.info("Generating embeddings...")
        embeddings = embed(sample_texts, prefix=EMBED_DOC_PREFIX)

        sample_data = [
            {"id": str(i), "content": text, "embedding": emb.tolist()}
            for i, (text, emb) in enumerate(zip(sample_texts, embeddings))
        ]

        logger.info("Inserting data...")
        client.bulk_ingest(table_name, sample_data, dimension=dimension)
        client.create_index(table_name, metric="cosine", method="hnsw")

        # Search Query
        query_text = "Python programming"
        logger.info(f"\nQuery: '{query_text}'")
        query_embedding = embed(query_text, prefix=EMBED_QUERY_PREFIX).tolist()

        # 1. Standard Semantic Search (Top 5)
        logger.info("\n--- Standard Semantic Search (Top 5) ---")
        standard_results = client.search(
            table_name=table_name,
            query_embedding=query_embedding,
            top_k=5,
            metric="cosine",
        )

        for res in standard_results:
            print(f"ID: {res['id']} | Score: {res['score']:.4f} | {res['content']}")

        # 2. Diverse Search (MMR)
        logger.info("\n--- Diverse Search (MMR, Top 5, Lambda=0.3) ---")
        diverse_results = client.diverse_search(
            table_name=table_name,
            query_embedding=query_embedding,
            top_k=5,
            num_candidates=10,
            lambda_mult=0.3,  # Lower lambda = More diversity
            metric="cosine",
        )

        for res in diverse_results:
            print(f"ID: {res['id']} | Content: {res['content']}")

    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        logger.info("\nCleaning up...")
        client.delete_db(confirm=True)
        client.close()


if __name__ == "__main__":
    main()
