import numpy as np
import uuid
from numpy.typing import NDArray
from psycopg import connect, sql, errors
from pgvector.psycopg import register_vector
from typing import List, Dict, Optional, Tuple, TypedDict, Union
from psycopg.rows import dict_row
from jet.db.postgres.scoring import calculate_vector_scores
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)

VectorInput = Union[List[float], NDArray[np.float64]]


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
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False
    ) -> None:
        """Ensure database exists, then connect and initialize."""
        self._ensure_database_exists(
            dbname, user, password, host, port, overwrite_db)
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

    def _to_list(self, vector: VectorInput) -> List[float]:
        """Convert vector input to list of floats."""
        if isinstance(vector, np.ndarray):
            return vector.tolist()
        return vector

    def _ensure_database_exists(self, dbname: str, user: str, password: str, host: str, port: int, overwrite_db: bool) -> None:
        """Drop the target database if it exists and overwrite_db is True, then create a new one."""
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
                if exists and overwrite_db:
                    # Terminate any active connections to the database before dropping
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                            "FROM pg_stat_activity "
                            "WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (dbname,)
                    )
                    # Drop the existing database
                    cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(dbname)))
                if not exists or overwrite_db:
                    # Create the new database
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(dbname)))

    def _initialize_extension(self) -> None:
        """Ensure the pgvector extension is enabled only if not already present."""
        with self.conn.cursor() as cur:
            try:
                cur.execute("SHOW dynamic_library_path;")
                libdir = cur.fetchone()["dynamic_library_path"]
                print(f"Debug: Dynamic library path: {libdir}")
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = 'vector';")
                exists = cur.fetchone()
                if not exists:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except errors.UndefinedFile as e:
                raise RuntimeError(
                    f"pgvector extension is not installed on the PostgreSQL server. "
                    f"Dynamic library path: {libdir}. "
                    f"Ensure vector.dylib is in /opt/homebrew/opt/postgresql@16/lib/postgresql. "
                    "Please install it using 'brew install pgvector' or compile from source: "
                    "https://github.com/pgvector/pgvector. "
                    f"Error: {str(e)}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize pgvector extension: {str(e)}"
                ) from e

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
        query = sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {} (
                id TEXT PRIMARY KEY,
                embedding VECTOR({})
            );
            """
        ).format(sql.Identifier(table_name), sql.Literal(dimension))
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

    def insert_vector(self, table_name: str, vector: VectorInput) -> str:
        """Insert a vector into the table with a hash-based ID."""
        vector_id = self.generate_unique_hash()
        vector_list = self._to_list(vector)
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, vector_list))
        return vector_id

    def insert_vectors(self, table_name: str, vectors: List[VectorInput]) -> List[str]:
        """Insert multiple vectors into the table and return their generated hash-based IDs."""
        vector_ids = [self.generate_unique_hash() for _ in vectors]
        formatted_vectors = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in vectors]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids, formatted_vectors))
        return vector_ids

    def insert_vector_by_id(self, table_name: str, vector_id: str, vector: VectorInput) -> None:
        """Insert a vector with a specific ID."""
        formatted_vector = f"[{', '.join(map(str, self._to_list(vector)))}]"
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s::vector);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, formatted_vector))

    def insert_vector_by_ids(self, table_name: str, vector_data: Dict[str, VectorInput]) -> None:
        """Insert multiple vectors with specific IDs."""
        ids = list(vector_data.keys())
        formatted_vectors = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in vector_data.values()]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, formatted_vectors))

    def update_vector_by_id(self, table_name: str, vector_id: str, new_vector: VectorInput) -> None:
        """Update a vector by its hash ID."""
        query = f"UPDATE {table_name} SET embedding = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (self._to_list(new_vector), vector_id))

    def update_vector_by_ids(self, table_name: str, updates: Dict[str, VectorInput]) -> None:
        """Update multiple vectors by their hash-based IDs."""
        ids = list(updates.keys())
        embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in updates.values()]
        query = f"UPDATE {table_name} SET embedding = data.embedding FROM (SELECT UNNEST(%s::text[]) AS id, UNNEST(%s::vector[]) AS embedding) AS data WHERE {table_name}.id = data.id;"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, embeddings))

    def search_similar(self, table_name: str, query_vector: VectorInput, top_k: int = 5) -> List[SearchResult]:
        """Find the top-K most similar vectors using L2 distance."""
        query = f"SELECT id, embedding <-> %s::vector as distance FROM {table_name} ORDER BY distance LIMIT {top_k};"
        formatted_vector = f"[{', '.join(map(str, self._to_list(query_vector)))}]"
        with self.conn.cursor() as cur:
            cur.execute(query, (formatted_vector,))
            db_results = cur.fetchall()
            results = [{"id": row["id"], "distance": row["distance"]}
                       for row in db_results]
        distances = [res["distance"] for res in results]
        scores = calculate_vector_scores(distances)
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
