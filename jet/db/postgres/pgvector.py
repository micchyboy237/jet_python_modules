from psycopg import connect, sql
from pgvector.psycopg import register_vector
import numpy as np
import uuid
from typing import List, Dict, Optional, Tuple, TypedDict
from psycopg.rows import dict_row
from jet.db.postgres.scoring import calculate_vector_scores
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)


class SearchResult(TypedDict):
    id: str
    score: float


class PgVectorClient:
    def __init__(
        self,
        dbname: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT
    ) -> None:
        """Ensure database exists, then connect and initialize."""
        self._ensure_database_exists(dbname, user, password, host, port)

        self.conn = connect(
            dbname=dbname,
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=False,
            row_factory=dict_row
        )
        self._initialize_extension()
        register_vector(self.conn)

    def _ensure_database_exists(self, dbname, user, password, host, port):
        """Connect to default DB and create the target DB if it doesn't exist."""
        with connect(
            dbname="postgres",
            user=user,
            password=password,
            host=host,
            port=port,
            autocommit=True
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
                exists = cur.fetchone()
                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(dbname)))

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

    def get_vectors(self, table_name: str) -> Optional[list[list[float]]]:
        """Retrieve all vectors."""
        query = f"SELECT embedding FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return [np.array(row["embedding"]).tolist() for row in results] if results else None

    def count_vectors(self, table_name: str) -> Optional[int]:
        """Count the total number of vectors in the table."""
        query = f"SELECT COUNT(*) FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            return result["count"] if result else None

    def get_vector_by_id(self, table_name: str, vector_id: str) -> Optional[list[float]]:
        """Retrieve a vector by its hash ID."""
        query = f"SELECT embedding FROM {table_name} WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id,))
            result = cur.fetchone()
            return np.array(result["embedding"]).tolist() if result else None

    def get_vectors_by_ids(self, table_name: str, vector_ids: List[str]) -> Dict[str, list[float]]:
        """Retrieve multiple vectors by their hash-based IDs."""
        query = f"SELECT id, embedding FROM {table_name} WHERE id = ANY(%s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids,))
            results = cur.fetchall()
            return {row["id"]: np.array(row["embedding"]).tolist() for row in results}

    def insert_vector(self, table_name: str, vector: List[float]) -> str:
        """Insert a vector into the table with a hash-based ID."""
        vector_id = self.generate_unique_hash()
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, vector))
        return vector_id

    def insert_vectors(self, table_name: str, vectors: List[List[float]]) -> List[str]:
        """Insert multiple vectors into the table and return their generated hash-based IDs."""
        vector_ids = [self.generate_unique_hash() for _ in vectors]
        formatted_vectors = [f"[{', '.join(map(str, v))}]" for v in vectors]

        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"

        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids, formatted_vectors))

        return vector_ids

    def insert_vector_by_id(self, table_name: str, vector_id: str, vector: List[float]) -> None:
        """Insert a vector with a specific ID."""
        formatted_vector = f"[{', '.join(map(str, vector))}]"

        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s::vector);"

        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, formatted_vector))

    def insert_vector_by_ids(self, table_name: str, vector_data: Dict[str, List[float]]) -> None:
        """Insert multiple vectors with specific IDs."""
        ids = list(vector_data.keys())
        formatted_vectors = [
            f"[{', '.join(map(str, v))}]" for v in vector_data.values()]

        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"

        with self.conn.cursor() as cur:
            cur.execute(query, (ids, formatted_vectors))

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

    def search_similar(self, table_name: str, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Find the top-K most similar vectors using L2 distance."""
        query = f"""
        SELECT id, embedding <-> %s::vector AS distance
        FROM {table_name}
        ORDER BY distance
        LIMIT {top_k};
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (query_vector,))
            db_results = cur.fetchall()
            results = [{"id": row["id"], "distance": row["distance"]}
                       for row in db_results]

        # Compute similarity scores
        distances = [res["distance"] for res in results]
        scores = calculate_vector_scores(distances)

        # Attach scores to results
        final_results: List[SearchResult] = []
        for res, score in zip(results, scores):
            final_results.append({
                "id": res["id"],
                "score": score
            })

        return sorted(final_results, key=lambda x: x['score'], reverse=True)

    def drop_all_rows(self, table_name: Optional[str] = None) -> None:
        """Delete all rows from a specific table or all tables if no table is provided."""
        with self.conn.cursor() as cur:
            if table_name:
                query = f"DELETE FROM {table_name};"
                cur.execute(query)
            else:
                # Get all user-defined table names
                cur.execute("""
                    SELECT tablename 
                    FROM pg_tables 
                    WHERE schemaname = 'public';
                """)
                tables = [row["tablename"] for row in cur.fetchall()]

                # Delete all rows from each table
                for table in tables:
                    cur.execute(f"DELETE FROM {table};")

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


__all__ = [
    "SearchResult",
    "PgVectorClient",
]
