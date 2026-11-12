# jet_python_modules/jet/db/postgres/examples/demo_filtered_search.py
from jet.adapters.llama_cpp.config import EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX
from jet.adapters.llama_cpp.embed_utils import embed
from jet.db.postgres.pgvector import PgVectorClient
from jet.logger import logger


def main():
    demo_db_name = "jet_vector_filter_demo_final"
    logger.info(f"Initializing PgVectorClient for database: {demo_db_name}")
    client = PgVectorClient(dbname=demo_db_name, overwrite_db=True)

    try:
        table_name = "filtered_items"
        dimension = 768

        client.create_vector_table(
            table_name,
            dimension,
            additional_columns={
                "content": "TEXT",
                "code": "TEXT",
                "genres": "JSONB",
                "release_date": "TIMESTAMPTZ",
                "director": "TEXT",  # Used for null checks
            },
        )

        # Sample data covering various scenarios
        sample_data = [
            {
                "id": "1",
                "content": "High speed car chase in Tokyo",
                "code": "ssis",
                "genres": ["Action", "Cars"],
                "release_date": "2023-05-15T00:00:00+00:00",
                "director": None,
            },
            {
                "id": "2",
                "content": "Romantic getaway in Paris",
                "code": "abp",
                "genres": ["Romance", "Travel"],
                "release_date": "2024-01-10T00:00:00+00:00",
                "director": "Jean Luc",
            },
            {
                "id": "3",
                "content": "Sci-fi battle in space",
                "code": "ssis",
                "genres": ["Sci-Fi", "Action"],
                "release_date": "2023-11-20T00:00:00+00:00",
                "director": None,
            },
            {
                "id": "4",
                "content": "Documentary about ocean life",
                "code": "doc",
                "genres": ["Documentary", "Nature"],
                "release_date": "2022-08-05T00:00:00+00:00",
                "director": "David Attenborough",
            },
        ]

        logger.info("Generating embeddings...")
        texts = [d["content"] for d in sample_data]
        embeddings = embed(texts, prefix=EMBED_DOC_PREFIX)

        for i, d in enumerate(sample_data):
            d["embedding"] = embeddings[i].tolist()

        client.bulk_ingest(table_name, sample_data, dimension=dimension)
        client.create_index(table_name, metric="cosine", method="hnsw")

        query_text = "Exciting action scene"
        query_emb = embed(query_text, prefix=EMBED_QUERY_PREFIX).tolist()

        # 1. Multiple Codes (Using $in)
        logger.info("\n--- 1. Filter by Multiple Codes: ['ssis'] ---")
        results = client.search(
            table_name,
            query_emb,
            top_k=5,
            filters={"code": {"$in": ["ssis"]}},
        )
        for r in results:
            print(f"ID: {r['id']} | Code: {r.get('code')}")

        # 2. Multiple Genres (Using $contains for JSONB)
        # Finds items that have BOTH Action AND Sci-Fi (if supported by @>) or at least one.
        # Note: @> checks if the left JSONB contains the right JSONB.
        logger.info("\n--- 2. Filter by Genres containing 'Action' ---")
        results = client.search(
            table_name,
            query_emb,
            top_k=5,
            filters={"genres": {"$contains": ["Action"]}},
        )
        for r in results:
            print(f"ID: {r['id']} | Genres: {r.get('genres')}")

        # 3. Multiple Fields are Null (Using standard None check)
        logger.info("\n--- 3. Filter where Director IS NULL ---")
        results = client.search(
            table_name, query_emb, top_k=5, filters={"director": None}
        )
        for r in results:
            print(f"ID: {r['id']} | Director: {r.get('director')}")

        # 4. Release Date Range (Using $gte and $lte)
        logger.info("\n--- 4. Filter by Release Date in 2023 ---")
        results = client.search(
            table_name,
            query_emb,
            top_k=5,
            filters={
                "release_date": {
                    "$gte": "2023-01-01T00:00:00+00:00",
                    "$lte": "2023-12-31T23:59:59+00:00",
                }
            },
        )
        for r in results:
            print(f"ID: {r['id']} | Release: {r.get('release_date')}")

        # 5. Complex Combined Filter
        logger.info(
            "\n--- 5. Combined: Code IN ['ssis', 'abp'] AND Genres contains 'Action' ---"
        )
        results = client.search(
            table_name,
            query_emb,
            top_k=5,
            filters={
                "code": {"$in": ["ssis", "abp"]},
                "genres": {"$contains": ["Sci-Fi"]},
            },
        )
        for r in results:
            print(f"ID: {r['id']} | Code: {r.get('code')} | Genres: {r.get('genres')}")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        client.delete_db(confirm=True)
        client.close()


if __name__ == "__main__":
    main()
