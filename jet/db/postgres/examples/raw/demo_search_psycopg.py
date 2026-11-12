import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

# Configuration
DB_CONFIG = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "",  # Replace with your actual password
    "host": "localhost",
    "port": 5432,
}


def main():
    # 1. Connect to PostgreSQL
    # Note: autocommit=True is used for DDL operations (CREATE/DROP)
    # For transactions, you might want to manage commits manually.
    conn = psycopg.connect(**DB_CONFIG, autocommit=True)

    try:
        # 2. Register pgvector types with psycopg
        # This allows psycopg to understand the 'vector' type in Postgres
        register_vector(conn)

        cursor = conn.cursor()

        # 3. Cleanup previous runs (Optional)
        print("Cleaning up existing table...")
        cursor.execute("DROP TABLE IF EXISTS items")

        # 4. Create Table
        # We define a vector column with 3 dimensions for this example
        print("Creating table 'items'...")
        cursor.execute("""
            CREATE TABLE items (
                id bigserial PRIMARY KEY,
                content text,
                embedding vector(3)
            )
        """)

        # 5. Insert Sample Data
        # In a real app, you would generate these embeddings using a model
        # like OpenAI, SentenceTransformers, or Hugging Face.
        print("Inserting sample data...")

        sample_data = [
            ("The dog is barking", [0.1, 0.2, 0.3]),
            ("The cat is purring", [0.2, 0.1, 0.4]),
            ("The bear is growling", [0.9, 0.8, 0.7]),
            ("Python programming is fun", [0.5, 0.5, 0.5]),
        ]

        for content, emb_list in sample_data:
            # Convert list to pgvector Vector object
            embedding = Vector(emb_list)
            cursor.execute(
                "INSERT INTO items (content, embedding) VALUES (%s, %s)",
                (content, embedding),
            )

        print(f"Inserted {len(sample_data)} items.")

        # 6. Perform Vector Search
        # Let's search for something similar to "The dog is barking"
        query_text = "A canine is making noise"
        # Mock embedding for the query (in reality, pass this through your embedding model)
        query_embedding = Vector([0.15, 0.25, 0.35])

        print(f"\nSearching for similarities to: '{query_text}'")
        print(f"Query Vector: {query_embedding.to_list()}")

        # The '<->' operator calculates Euclidean distance (L2 distance)
        # For Cosine Distance, use '<=>'
        # For Inner Product, use '<#>'
        cursor.execute(
            """
            SELECT id, content, embedding <-> %s AS distance
            FROM items
            ORDER BY embedding <-> %s
            LIMIT 3
        """,
            (query_embedding, query_embedding),
        )

        results = cursor.fetchall()

        print("\nTop 3 Results (Euclidean Distance):")
        print("-" * 40)
        for row in results:
            item_id, content, distance = row
            print(f"ID: {item_id} | Content: {content} | Distance: {distance:.4f}")

        # 7. Demonstrate Cosine Similarity Search
        print("\nTop 3 Results (Cosine Distance):")
        print("-" * 40)
        cursor.execute(
            """
            SELECT id, content, embedding <=> %s AS cosine_distance
            FROM items
            ORDER BY embedding <=> %s
            LIMIT 3
        """,
            (query_embedding, query_embedding),
        )

        results_cosine = cursor.fetchall()
        for row in results_cosine:
            item_id, content, distance = row
            print(f"ID: {item_id} | Content: {content} | Cosine Dist: {distance:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        # 8. Final Cleanup (Drop table after demo)
        print("\nCleaning up database...")
        try:
            cursor.execute("DROP TABLE IF EXISTS items")
            print("Table 'items' dropped successfully.")
        except Exception as e:
            print(f"Cleanup error: {e}")
        finally:
            cursor.close()
            conn.close()
            print("Connection closed.")


if __name__ == "__main__":
    main()
