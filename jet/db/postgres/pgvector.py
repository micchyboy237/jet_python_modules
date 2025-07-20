import numpy as np
import uuid
from numpy.typing import NDArray
from psycopg import connect, sql, errors
from pgvector.psycopg import register_vector
from typing import List, Dict, Optional, Tuple, TypedDict, Union
from psycopg.rows import dict_row
from jet.db.postgres.scoring import calculate_vector_scores
from jet.logger import logger
from .config import (
    DEFAULT_DB,
    DEFAULT_USER,
    DEFAULT_PASSWORD,
    DEFAULT_HOST,
    DEFAULT_PORT,
)

Embedding = NDArray[np.float64]
EmbeddingInput = Union[List[float], Embedding]


class SearchResult(TypedDict):
    id: str
    score: float


class DatabaseMetadata(TypedDict):
    dbname: str
    owner: str
    encoding: str
    collation: Optional[str]
    ctype: Optional[str]
    size_mb: float


class ColumnMetadata(TypedDict):
    column_name: str
    data_type: str
    is_nullable: str
    character_maximum_length: Optional[int]
    numeric_precision: Optional[int]
    numeric_scale: Optional[int]


class TableMetadata(TypedDict):
    table_name: str
    table_type: str
    schema_name: str
    row_count: int
    columns: List[ColumnMetadata]


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

    def _to_list(self, embedding: EmbeddingInput) -> List[float]:
        """Convert embedding input to list of floats."""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

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
                    cur.execute(
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                            "FROM pg_stat_activity "
                            "WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (dbname,)
                    )
                    cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(dbname)))
                if not exists or overwrite_db:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(dbname)))

    def _initialize_extension(self) -> None:
        """Ensure the pgvector extension is enabled only if not already present."""
        with self.conn.cursor() as cur:
            try:
                cur.execute("SHOW dynamic_library_path;")
                libdir = cur.fetchone()["dynamic_library_path"]
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

    def _ensure_table_exists(self, table_name: str, dimension: int) -> None:
        """Check if table exists, create it with the specified dimension if it doesn't."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_tables WHERE schemaname = 'public' AND tablename = %s;",
                (table_name,)
            )
            table_exists = cur.fetchone()
            if not table_exists:
                self.create_table(table_name, dimension)

    def __enter__(self):
        """Begin transaction when entering 'with' block."""
        self.conn.execute("BEGIN;")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Handle commit or rollback when exiting 'with' block."""
        if self.conn is None or self.conn.closed:
            return
        if exc_type is None:
            self.conn.execute("COMMIT;")
        else:
            self.conn.execute("ROLLBACK;")

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

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
        """Create a table with an embedding column using hash-based IDs."""
        query = sql.SQL(
            "CREATE TABLE IF NOT EXISTS {} ("
            "id TEXT PRIMARY KEY, "
            "embedding vector({})"
            ");"
        ).format(sql.Identifier(table_name), sql.Literal(dimension))
        with self.conn.cursor() as cur:
            cur.execute(query)

    def generate_unique_hash(self) -> str:
        """Generate a unique UUID v4 string."""
        return str(uuid.uuid4())

    def get_rows(self, table_name: str, ids: Optional[List[str]] = None) -> List[Dict[str, Union[str, Embedding]]]:
        """Retrieve rows from the specified table, optionally filtered by IDs.

        Args:
            table_name: Name of the table to query
            ids: Optional list of IDs to filter results

        Returns:
            List of dictionaries containing id and embedding for each row
        """
        query = f"SELECT id, embedding FROM {table_name}"
        params = ()
        if ids is not None:
            query += " WHERE id = ANY(%s)"
            params = (ids,)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            return [
                {
                    "id": row["id"],
                    "embedding": np.array(row["embedding"])
                }
                for row in results
            ]

    def get_embeddings(self, table_name: str, ids: Optional[List[str]] = None) -> Dict[str, Embedding]:
        """Retrieve embeddings with their IDs from the specified table, optionally filtered by IDs.

        Args:
            table_name: Name of the table to query
            ids: Optional list of IDs to filter results

        Returns:
            Dictionary mapping embedding IDs to their corresponding embeddings
        """
        query = f"SELECT id, embedding FROM {table_name}"
        params = ()
        if ids is not None:
            query += " WHERE id = ANY(%s)"
            params = (ids,)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            return {row["id"]: np.array(row["embedding"]) for row in results}

    def count_embeddings(self, table_name: str) -> Optional[int]:
        """Count the total number of embeddings in the table."""
        query = f"SELECT COUNT(*) FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            return result["count"] if result else None

    def get_embedding_by_id(self, table_name: str, embedding_id: str) -> Optional[NDArray[np.float64]]:
        """Retrieve an embedding by its hash ID."""
        query = f"SELECT embedding FROM {table_name} WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id,))
            result = cur.fetchone()
            return np.array(result["embedding"]) if result else None

    def insert_embedding(self, table_name: str, embedding: EmbeddingInput, dimension: Optional[int] = None) -> str:
        """Insert an embedding into the table with a hash-based ID, creating the table if needed."""
        if dimension is None:
            dimension = len(self._to_list(embedding))
        self._ensure_table_exists(table_name, dimension)
        embedding_id = self.generate_unique_hash()
        embedding_list = self._to_list(embedding)
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, embedding_list))
        return embedding_id

    def insert_embeddings(self, table_name: str, embeddings: List[EmbeddingInput], dimension: Optional[int] = None) -> List[str]:
        """Insert multiple embeddings into the table and return their generated hash-based IDs, creating the table if needed."""
        if not embeddings:
            raise ValueError("Cannot insert empty embedding list")
        if dimension is None:
            dimension = len(self._to_list(embeddings[0]))
        self._ensure_table_exists(table_name, dimension)
        embedding_ids = [self.generate_unique_hash() for _ in embeddings]
        formatted_embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in embeddings]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_ids, formatted_embeddings))
        return embedding_ids

    def insert_embedding_by_id(self, table_name: str, embedding_id: str, embedding: EmbeddingInput, dimension: Optional[int] = None) -> None:
        """Insert an embedding with a specific ID, creating the table if limited to one change per filetable if needed."""
        if dimension is None:
            dimension = len(self._to_list(embedding))
        self._ensure_table_exists(table_name, dimension)
        formatted_embedding = f"[{', '.join(map(str, self._to_list(embedding)))}]"
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s::vector);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, formatted_embedding))

    def insert_embedding_by_ids(self, table_name: str, embedding_data: Dict[str, EmbeddingInput], dimension: Optional[int] = None) -> None:
        """Insert multiple embeddings with specific IDs, creating the table if needed."""
        if not embedding_data:
            raise ValueError("Cannot insert empty embedding data")
        if dimension is None:
            dimension = len(self._to_list(next(iter(embedding_data.values()))))
        self._ensure_table_exists(table_name, dimension)
        ids = list(embedding_data.keys())
        formatted_embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in embedding_data.values()]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, formatted_embeddings))

    def update_embedding_by_id(self, table_name: str, embedding_id: str, new_embedding: EmbeddingInput) -> None:
        """Update an embedding by its hash ID."""
        query = f"UPDATE {table_name} SET embedding = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (self._to_list(new_embedding), embedding_id))

    def update_embedding_by_ids(self, table_name: str, updates: Dict[str, EmbeddingInput]) -> None:
        """Update multiple embeddings by their hash-based IDs."""
        ids = list(updates.keys())
        embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in updates.values()]
        query = f"UPDATE {table_name} SET embedding = data.embedding FROM (SELECT UNNEST(%s::text[]) AS id, UNNEST(%s::vector[]) AS embedding) AS data WHERE {table_name}.id = data.id;"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, embeddings))

    def search_similar(
        self,
        table_name: str,
        query_embedding: EmbeddingInput,
        top_k: int = 5
    ) -> List[SearchResult]:
        query = f"SELECT id, embedding <=> %s::vector as distance FROM {table_name} ORDER BY distance LIMIT {top_k};"
        formatted_embedding = f"[{', '.join(map(str, self._to_list(query_embedding)))}]"
        with self.conn.cursor() as cur:
            cur.execute(query, (formatted_embedding,))
            db_results = cur.fetchall()
            results = [{"id": row["id"], "distance": row["distance"]}
                       for row in db_results]
        distances = [res["distance"] for res in results]
        logger.debug("Raw cosine distances from %s: %s", table_name, distances)
        scores = calculate_vector_scores(distances)
        logger.debug("Transformed similarity scores: %s", scores)
        final_results: List[SearchResult] = []
        for res, score in zip(results, scores):
            final_results.append({
                "id": res["id"],
                "score": score
            })
        return final_results

    def drop_all_rows(self, table_name: Optional[str] = None) -> None:
        """Delete all rows from a specific table or all tables if no table is provided."""
        with self.conn.cursor() as cur:
            if table_name:
                query = f"DELETE FROM {table_name};"
                cur.execute(query)
            else:
                cur.execute(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
                tables = [row["tablename"] for row in cur.fetchall()]
                for table in tables:
                    cur.execute(f"DELETE FROM {table};")

    def delete_all_tables(self) -> None:
        """Drop all tables in the database."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public';")
            tables = [row["tablename"] for row in cur.fetchall()]
            for table in tables:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(
                    sql.Identifier(table)))

    def delete_db(self, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError(
                "Database deletion requires explicit confirmation by setting confirm=True")
        dbname = self.conn.info.dbname
        user = self.conn.info.user
        password = self.conn.info.password
        host = self.conn.info.host
        port = self.conn.info.port
        if not self.conn.closed:
            self.close()
        try:
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
                        sql.SQL(
                            "SELECT pg_terminate_backend(pg_stat_activity.pid) "
                            "FROM pg_stat_activity "
                            "WHERE pg_stat_activity.datname = %s AND pid <> pg_backend_pid();"
                        ),
                        (dbname,)
                    )
                    cur.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(
                        sql.Identifier(dbname)))
        except Exception as e:
            raise RuntimeError(
                f"Failed to delete database {dbname}: {str(e)}") from e
        self.conn = None

    def get_database_metadata(self) -> DatabaseMetadata:
        """Retrieve metadata for the current database."""
        with self.conn.cursor() as cur:
            query = (
                "SELECT d.datname AS dbname, d.datdba::regrole::text AS owner, "
                "pg_encoding_to_char(d.encoding) AS encoding, d.datcollate AS collation, "
                "d.datctype AS ctype, "
                "pg_database_size(d.datname)::float / 1024 / 1024 AS size_mb "
                "FROM pg_database d WHERE d.datname = %s;"
            )
            cur.execute(query, (self.conn.info.dbname,))
            result = cur.fetchone()
            if not result:
                raise RuntimeError(
                    f"Database {self.conn.info.dbname} not found")
            return {
                "dbname": result["dbname"],
                "owner": result["owner"],
                "encoding": result["encoding"],
                "collation": result["collation"],
                "ctype": result["ctype"],
                "size_mb": round(result["size_mb"], 2),
            }

    def get_all_tables(self) -> List[str]:
        """Retrieve a list of all table names in the public schema."""
        with self.conn.cursor() as cur:
            query = (
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public';"
            )
            cur.execute(query)
            results = cur.fetchall()
            return [row["table_name"] for row in results]

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        """Retrieve detailed metadata for a specific table."""
        with self.conn.cursor() as cur:
            table_query = (
                "SELECT t.table_name, t.table_type, t.table_schema AS schema_name, "
                "COALESCE((SELECT reltuples::bigint FROM pg_class "
                "WHERE relname = t.table_name), 0) AS row_count "
                "FROM information_schema.tables t "
                "WHERE t.table_schema = 'public' AND t.table_name = %s;"
            )
            formatted_table_query = sql.SQL(table_query).format(
                table_name=sql.Identifier(table_name))
            cur.execute(formatted_table_query, (table_name,))
            table_result = cur.fetchone()
            if not table_result:
                raise RuntimeError(
                    f"Table {table_name} not found in public schema")
            column_query = (
                "SELECT column_name, data_type, is_nullable, "
                "character_maximum_length, numeric_precision, numeric_scale "
                "FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = %s;"
            )
            cur.execute(column_query, (table_name,))
            columns = [{
                "column_name": row["column_name"],
                "data_type": row["data_type"],
                "is_nullable": row["is_nullable"],
                "character_maximum_length": row["character_maximum_length"],
                "numeric_precision": row["numeric_precision"],
                "numeric_scale": row["numeric_scale"],
            } for row in cur.fetchall()]
            return {
                "table_name": table_result["table_name"],
                "table_type": table_result["table_type"],
                "schema_name": table_result["schema_name"],
                "row_count": table_result["row_count"],
                "columns": columns,
            }

    def get_database_summary(self) -> Dict[str, Union[DatabaseMetadata, List[TableMetadata]]]:
        """Retrieve a comprehensive summary of the database including metadata and all tables."""
        tables = self.get_all_tables()
        table_metadata = [self.get_table_metadata(table) for table in tables]
        return {
            "database_metadata": self.get_database_metadata(),
            "tables": table_metadata,
        }


__all__ = [
    "SearchResult",
    "PgVectorClient",
]
