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

VectorInput = Union[List[float], NDArray[np.float64]]


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

    def insert_vector(self, table_name: str, vector: VectorInput, dimension: Optional[int] = None) -> str:
        """Insert a vector into the table with a hash-based ID, creating the table if needed."""
        if dimension is None:
            dimension = len(self._to_list(vector))
        self._ensure_table_exists(table_name, dimension)
        vector_id = self.generate_unique_hash()
        vector_list = self._to_list(vector)
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, vector_list))
        return vector_id

    def insert_vectors(self, table_name: str, vectors: List[VectorInput], dimension: Optional[int] = None) -> List[str]:
        """Insert multiple vectors into the table and return their generated hash-based IDs, creating the table if needed."""
        if not vectors:
            raise ValueError("Cannot insert empty vector list")
        if dimension is None:
            dimension = len(self._to_list(vectors[0]))
        self._ensure_table_exists(table_name, dimension)
        vector_ids = [self.generate_unique_hash() for _ in vectors]
        formatted_vectors = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in vectors]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_ids, formatted_vectors))
        return vector_ids

    def insert_vector_by_id(self, table_name: str, vector_id: str, vector: VectorInput, dimension: Optional[int] = None) -> None:
        """Insert a vector with a specific ID, creating the table if needed."""
        if dimension is None:
            dimension = len(self._to_list(vector))
        self._ensure_table_exists(table_name, dimension)
        formatted_vector = f"[{', '.join(map(str, self._to_list(vector)))}]"
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s::vector);"
        with self.conn.cursor() as cur:
            cur.execute(query, (vector_id, formatted_vector))

    def insert_vector_by_ids(self, table_name: str, vector_data: Dict[str, VectorInput], dimension: Optional[int] = None) -> None:
        """Insert multiple vectors with specific IDs, creating the table if needed."""
        if not vector_data:
            raise ValueError("Cannot insert empty vector data")
        if dimension is None:
            dimension = len(self._to_list(next(iter(vector_data.values()))))
        self._ensure_table_exists(table_name, dimension)
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

    def search_similar(
        self,
        table_name: str,
        query_vector: VectorInput,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Find the top-K most similar vectors using cosine similarity.

        Args:
            table_name: Name of the table containing vectors
            query_vector: Query vector for similarity search
            top_k: Number of top results to return

        Returns:
            List of SearchResult dictionaries with IDs and similarity scores in [0, 1]
        """
        query = f"SELECT id, embedding <=> %s::vector as distance FROM {table_name} ORDER BY distance LIMIT {top_k};"
        formatted_vector = f"[{', '.join(map(str, self._to_list(query_vector)))}]"
        with self.conn.cursor() as cur:
            cur.execute(query, (formatted_vector,))
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

        # No need to sort, as database ORDER BY distance ensures correct order
        # and calculate_vector_scores preserves this (smaller distance -> higher score)
        return final_results

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

    def delete_db(self, confirm: bool = False) -> None:
        """Drop the entire database after terminating active connections, requires explicit confirmation.

        Args:
            confirm: Must be True to proceed with deletion to prevent accidental database drops.

        Raises:
            ValueError: If confirm is False.
            RuntimeError: If database deletion fails.
        """
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
        query = """
            SELECT 
                d.datname AS dbname,
                r.rolname AS owner,
                pg_encoding_to_char(d.encoding) AS encoding,
                d.datcollate AS collation,
                d.datctype AS ctype,
                (pg_database_size(d.datname) / 1024.0 / 1024.0) AS size_mb
            FROM pg_database d
            JOIN pg_roles r ON d.datdba = r.oid
            WHERE d.datname = %s;
        """
        with self.conn.cursor() as cur:
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
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """
        with self.conn.cursor() as cur:
            cur.execute(query)
            results = cur.fetchall()
            return [row["table_name"] for row in results]

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        """Retrieve detailed metadata for a specific table."""
        # Get table info
        table_query = """
            SELECT 
                t.table_name,
                t.table_type,
                t.table_schema AS schema_name,
                (SELECT COUNT(*) FROM {table_name}) AS row_count
            FROM information_schema.tables t
            WHERE t.table_schema = 'public' AND t.table_name = %s;
        """
        # Get column info
        column_query = """
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale
            FROM information_schema.columns c
            WHERE c.table_schema = 'public' AND c.table_name = %s
            ORDER BY c.ordinal_position;
        """
        with self.conn.cursor() as cur:
            # Execute table query with proper parameter handling
            formatted_table_query = sql.SQL(table_query).format(
                table_name=sql.Identifier(table_name))
            cur.execute(formatted_table_query, (table_name,))
            table_result = cur.fetchone()
            if not table_result:
                raise RuntimeError(
                    f"Table {table_name} not found in public schema")

            # Execute column query
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
