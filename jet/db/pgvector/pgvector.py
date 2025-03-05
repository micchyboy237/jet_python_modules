import psycopg
from pgvector.psycopg import register_vector
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple, TypedDict
from psycopg.rows import dict_row, tuple_row


class VectorRecord(TypedDict):
    id: str
    embedding: np.ndarray


class PgVectorClient:
    def __init__(self, dbname: str, user: str, password: str, host: str = 'localhost', port: int = 5432) -> None:
        """Initialize the database connection."""
        self.conn = psycopg.connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=False,  # Enable manual transaction control
            row_factory=dict_row
        )
        register_vector(self.conn)
        self._initialize_extension()

    def _initialize_extension(self) -> None:
        """Ensure the pgvector extension is enabled."""
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    def __enter__(self):
        """Begin transaction when entering 'with' block."""
        self.conn.execute("BEGIN;")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Handle commit or rollback when exiting 'with' block."""
        if exc_type is None:
            self.conn.execute("COMMIT;")
        else:
            self.conn.execute("ROLLBACK;")

    def begin_transaction(self) -> None:
        """Explicitly begin a new transaction."""
        self.conn.execute("BEGIN;")

    def commit(self) -> None:
        """Commit the current transaction."""
        self.conn.execute("COMMIT;")

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.conn.execute("ROLLBACK;")

    def create_table(self, table_name: str, dimension: int) -> None:
        """Create a table with a vector column using hash-based IDs."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            embedding VECTOR({dimension})
        );
        """
        with self.conn.cursor() as cur:
            cur.execute(query)

    def generate_unique_hash(self) -> str:
        """Generate a unique UUID v4 string."""
        return str(uuid.uuid4())

    def insert_vector(self, table_name: str, vector: List[float]) -> str:
        """Insert a vector into the table with a hash-based ID."""
        vector_id = self.generate_unique_hash()
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, vector))
        return vector_id

    def get_vector_by_id(self, table_name: str, vector_id: str) -> Optional[np.ndarray]:
        """Retrieve a vector by its hash ID."""
        query = f"SELECT embedding FROM {table_name} WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id,))
            result = cur.fetchone()
            return np.array(result["embedding"]) if result else None

    def get_vectors_by_ids(self, table_name: str, vector_ids: List[str]) -> Dict[str, np.ndarray]:
        """Retrieve multiple vectors by their hash-based IDs."""
        query = f"SELECT id, embedding FROM {table_name} WHERE id = ANY(%s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids,))
            results = cur.fetchall()
            return {row["id"]: np.array(row["embedding"]) for row in results}

    def update_vector_by_id(self, table_name: str, vector_id: str, new_vector: List[float]) -> None:
        """Update a vector by its hash ID."""
        query = f"UPDATE {table_name} SET embedding = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (new_vector, vector_id))

    def update_vector_by_ids(self, table_name: str, updates: Dict[str, List[float]]) -> None:
        """Update multiple vectors by their hash-based IDs."""
        query = f"""
        UPDATE {table_name} 
        SET embedding = data.embedding
        FROM (SELECT UNNEST(%s::text[]) AS id, UNNEST(%s::text[])::vector AS embedding) AS data 
        WHERE {table_name}.id = data.id;
        """

        ids = list(updates.keys())
        # Convert list to proper vector format
        embeddings = [f"[{', '.join(map(str, v))}]" for v in updates.values()]

        with self.conn.cursor() as cur:
            cur.execute(query, (ids, embeddings))

    def search_similar(self, table_name: str, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float]]:
        """Find the top-K most similar vectors using L2 distance."""
        query = f"""
        SELECT id, embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {top_k};
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (query_vector,))
            return [(row["id"], row["distance"]) for row in cur.fetchall()]

    def drop_all_rows(self, table_name: str) -> None:
        """Delete all rows from a table."""
        query = f"DELETE FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)

    def delete_all_tables(self) -> None:
        """Drop all tables in the database."""
        query = """
        DO $$ DECLARE 
            r RECORD;
        BEGIN 
            FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') 
            LOOP 
                EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
            END LOOP;
        END $$;
        """
        with self.conn.cursor() as cur:
            cur.execute(query)

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
