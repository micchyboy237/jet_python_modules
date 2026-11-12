# jet_python_modules/jet/db/postgres/examples/demo_vector_search_jet.py
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    demo_db_name = "jet_vector_demo"

    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=True)

    try:
        table_name = "demo_items"
        # nomic-embed typically uses 768 dimensions
        dimension = 768

        logger.info(
            f"Creating vector table '{table_name}' with dimension {dimension}..."
        )
        client.create_vector_table(
            table_name=table_name,
            dimension=dimension,
            additional_columns={"content": "TEXT"},
        )

        # Sample data with real-world concepts
        sample_texts = [
            "The dog is barking loudly at the mailman.",
            "The cat is purring softly on the windowsill.",
            "Python is a versatile programming language for data science.",
            "PostgreSQL is a powerful open-source relational database.",
            "The bear is growling in the forest during winter.",
        ]

        logger.info("Generating real embeddings using llama.cpp adapter...")
        # embed() returns a numpy array by default
        embeddings = embed(sample_texts)

        sample_data = [
            {"id": str(i + 1), "content": text, "embedding": emb.tolist()}
            for i, (text, emb) in enumerate(zip(sample_texts, embeddings))
        ]

        logger.info("Inserting sample data...")
        created_rows = client.create_rows(table_name, sample_data)
        logger.success(f"Inserted {len(created_rows)} items.")

        logger.info("Creating HNSW index for Cosine similarity...")
        client.create_index(table_name, metric="cosine", method="hnsw")

        logger.info("Creating Full-Text Search index...")
        client.create_fts_index(table_name, column_name="content")

        # Generate a real query embedding
        query_text = "A canine is making noise"
        logger.info(f"\nGenerating embedding for query: '{query_text}'")
        query_embedding = embed(query_text).tolist()

        logger.info(f"Searching for similarities...")

        results_cosine = client.search(
            table_name=table_name,
            query_embedding=query_embedding,
            top_k=3,
            metric="cosine",
        )

        print("\nTop 3 Results (Cosine Distance):")
        print("-" * 50)
        for res in results_cosine:
            print(
                f"ID: {res['id']} | Content: {res['content'][:40]}... | Score: {res['score']:.4f}"
            )

        logger.info("\nPerforming Hybrid Search...")
        hybrid_results = client.hybrid_search(
            table_name=table_name,
            query_text="dog barking",
            query_embedding=query_embedding,
            top_k=3,
            text_weight=1.0,
            vector_weight=1.0,
        )

        print("\nTop 3 Results (Hybrid - RRF):")
        print("-" * 50)
        for res in hybrid_results:
            print(
                f"ID: {res['id']} | Content: {res['content'][:40]}... | Hybrid Score: {res['hybrid_score']:.4f}"
            )

    except Exception as e:
        logger.error(f"An error occurred during demo: {e}")
    finally:
        logger.info("\nCleaning up database...")
        client.delete_db(confirm=True)
        logger.success("Demo database dropped successfully.")


if __name__ == "__main__":
    main()
