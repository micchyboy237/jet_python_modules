from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from pgvector import Vector
from psycopg import errors, sql

from jet.db.postgres.pg_types import EmbeddingInput


class VectorEngine:
    """Handles pgvector specific operations."""

    def __init__(self, conn):
        self.conn = conn
        self._initialize_extension()

    def _to_vector(self, embedding: EmbeddingInput) -> Vector:
        if isinstance(embedding, np.ndarray):
            return Vector(embedding.tolist())
        if isinstance(embedding, list):
            return Vector(embedding)
        return embedding

    def _initialize_extension(self):
        with self.conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except errors.UndefinedFile as e:
                raise RuntimeError("pgvector extension not installed") from e

    def create_index(
        self,
        table_name: str,
        index_name: Optional[str] = None,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
        method: Literal["hnsw", "ivfflat"] = "hnsw",
    ):
        ops_map = {
            "cosine": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "ip": "vector_ip_ops",
        }
        op_class = ops_map[metric]

        if not index_name:
            index_name = f"{table_name}_embedding_{metric}_{method}_idx"

        with self.conn.cursor() as cur:
            if method == "hnsw":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (embedding {})"
                    ).format(
                        sql.Identifier(index_name),
                        sql.Identifier(table_name),
                        sql.SQL(op_class),
                    )
                )
            elif method == "ivfflat":
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING ivfflat (embedding {}) WITH (lists = 100)"
                    ).format(
                        sql.Identifier(index_name),
                        sql.Identifier(table_name),
                        sql.SQL(op_class),
                    )
                )

    def get_embeddings(
        self, table_name: str, ids: Optional[List[str]] = None
    ) -> Dict[str, NDArray[np.float64]]:
        query = sql.SQL("SELECT id, embedding FROM {}").format(
            sql.Identifier(table_name)
        )
        params = ()
        if ids is not None:
            query = sql.SQL("{} WHERE id = ANY(%s)").format(query)
            params = (ids,)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            return {row["id"]: np.array(row["embedding"]) for row in results}

    def count_embeddings(self, table_name: str) -> int:
        query = sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table_name))
        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            return result["count"] if result else 0

    def get_embedding_by_id(
        self, table_name: str, embedding_id: str
    ) -> Optional[NDArray[np.float64]]:
        query = sql.SQL("SELECT embedding FROM {} WHERE id = %s;").format(
            sql.Identifier(table_name)
        )
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id,))
            result = cur.fetchone()
            return np.array(result["embedding"]) if result else None

    def insert_embedding(
        self,
        table_name: str,
        embedding: EmbeddingInput,
        dimension: Optional[int] = None,
    ) -> str:
        import uuid

        vec = self._to_vector(embedding)
        if dimension is None:
            dimension = len(vec)

        embedding_id = str(uuid.uuid4())
        query = sql.SQL("INSERT INTO {} (id, embedding) VALUES (%s, %s);").format(
            sql.Identifier(table_name)
        )

        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, vec))
        return embedding_id

    def insert_embeddings(
        self,
        table_name: str,
        embeddings: List[EmbeddingInput],
        dimension: Optional[int] = None,
    ) -> List[str]:
        import uuid

        if not embeddings:
            raise ValueError("Cannot insert empty embedding list")

        vecs = [self._to_vector(e) for e in embeddings]
        if dimension is None:
            dimension = len(vecs[0])

        embedding_ids = [str(uuid.uuid4()) for _ in embeddings]

        query = sql.SQL(
            "INSERT INTO {} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        ).format(sql.Identifier(table_name))

        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_ids, vecs))
        return embedding_ids

    def insert_embedding_by_id(
        self,
        table_name: str,
        embedding_id: str,
        embedding: EmbeddingInput,
        dimension: Optional[int] = None,
    ) -> None:
        vec = self._to_vector(embedding)
        if dimension is None:
            dimension = len(vec)

        query = sql.SQL("INSERT INTO {} (id, embedding) VALUES (%s, %s);").format(
            sql.Identifier(table_name)
        )
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, vec))

    def insert_embeddings_by_ids(
        self,
        table_name: str,
        embedding_data: Dict[str, EmbeddingInput],
        dimension: Optional[int] = None,
    ) -> None:
        if not embedding_data:
            raise ValueError("Cannot insert empty embedding data")

        ids = list(embedding_data.keys())
        vecs = [self._to_vector(v) for v in embedding_data.values()]

        if dimension is None:
            dimension = len(vecs[0])

        query = sql.SQL(
            "INSERT INTO {} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        ).format(sql.Identifier(table_name))

        with self.conn.cursor() as cur:
            cur.execute(query, (ids, vecs))

    def update_embedding_by_id(
        self, table_name: str, embedding_id: str, new_embedding: EmbeddingInput
    ) -> None:
        vec = self._to_vector(new_embedding)
        query = sql.SQL("UPDATE {} SET embedding = %s WHERE id = %s;").format(
            sql.Identifier(table_name)
        )
        with self.conn.cursor() as cur:
            cur.execute(query, (vec, embedding_id))

    def update_embedding_by_ids(
        self, table_name: str, updates: Dict[str, EmbeddingInput]
    ) -> None:
        ids = list(updates.keys())
        vecs = [self._to_vector(v) for v in updates.values()]

        query = sql.SQL(
            "UPDATE {} SET embedding = data.embedding FROM (SELECT UNNEST(%s::text[]) AS id, UNNEST(%s::vector[]) AS embedding) AS data WHERE {}.id = data.id;"
        ).format(sql.Identifier(table_name), sql.Identifier(table_name))

        with self.conn.cursor() as cur:
            cur.execute(query, (ids, vecs))

    def bulk_ingest(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        dimension: Optional[int] = None,
        batch_size: int = 1000,
    ) -> int:
        """
        Efficiently ingest a large list of records using PostgreSQL COPY.

        Args:
            table_name: Name of the target table.
            data: List of dicts. Each dict must have 'id', 'content', and 'embedding'.
            dimension: Optional vector dimension check.
            batch_size: Number of rows to process in memory at once.

        Returns:
            Total number of rows inserted.
        """
        if not data:
            return 0

        import io

        total_inserted = 0

        # Prepare columns based on the first item
        columns = list(data[0].keys())
        if "embedding" not in columns:
            raise ValueError("Data must contain an 'embedding' key")

        with self.conn.cursor() as cur:
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]

                # Create a CSV-like string buffer for COPY
                # We use a tab-separated format which is standard for COPY
                buffer = io.StringIO()

                for row in batch:
                    # Ensure embedding is a vector string format [1.2, 3.4]
                    emb = row["embedding"]
                    if isinstance(emb, np.ndarray):
                        emb_str = f"[{','.join(map(str, emb.tolist()))}]"
                    elif isinstance(emb, list):
                        emb_str = f"[{','.join(map(str, emb))}]"
                    else:
                        emb_str = str(emb)

                    # Map other columns
                    vals = []
                    for col in columns:
                        val = row.get(col)
                        if val is None:
                            vals.append("\\N")  # NULL in COPY
                        elif col == "embedding":
                            vals.append(emb_str)
                        else:
                            # Escape tabs and newlines for COPY format
                            s = (
                                str(val)
                                .replace("\\", "\\\\")
                                .replace("\n", "\\n")
                                .replace("\t", "\\t")
                            )
                            vals.append(s)

                    buffer.write("\t".join(vals) + "\n")

                buffer.seek(0)

                # Execute COPY
                col_names = sql.SQL(", ").join(map(sql.Identifier, columns))
                copy_sql = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT text)").format(
                    sql.Identifier(table_name), col_names
                )

                with cur.copy(copy_sql) as copy:
                    while True:
                        chunk = buffer.read(8192)
                        if not chunk:
                            break
                        copy.write(chunk)

                total_inserted += len(batch)

        return total_inserted
