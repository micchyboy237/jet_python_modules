import json
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray
from psycopg import errors, sql

from jet.db.postgres.pg_types import EmbeddingInput
from jet.db.postgres.scoring import calculate_vector_scores


class VectorEngine:
    """Handles pgvector specific operations."""

    def __init__(self, conn):
        self.conn = conn
        self._initialize_extension()

    def _to_list(self, embedding: EmbeddingInput) -> list[float]:
        """Convert embedding input to list of floats."""
        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
        return embedding

    def _initialize_extension(self):
        with self.conn.cursor() as cur:
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except errors.UndefinedFile as e:
                raise RuntimeError("pgvector extension not installed") from e

    def search_similar(
        self,
        table_name: str,
        query_vector: List[float],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        formatted_vec = f"[{', '.join(map(str, query_vector))}]"
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))
            query = sql.SQL(
                "SELECT {}, embedding <=> %s::vector as distance FROM {} ORDER BY distance LIMIT %s"
            ).format(select_cols, sql.Identifier(table_name))

            cur.execute(query, (formatted_vec, top_k))
            results = cur.fetchall()

            final_results = []
            distances = [r["distance"] for r in results]
            scores = calculate_vector_scores(distances)

            for rank, (res, score) in enumerate(zip(results, scores), start=1):
                if threshold and score < threshold:
                    continue
                entry = {
                    "id": res["id"],
                    "rank": rank,
                    "score": score,
                    "distance": res["distance"],
                }
                for c in cols:
                    if c != "id":
                        val = res[c]
                        if (
                            isinstance(val, str)
                            and val.startswith("{")
                            and val.endswith("}")
                        ):
                            try:
                                val = json.loads(val)
                            except:
                                pass
                        entry[c] = val
                final_results.append(entry)
            return final_results

    def get_embeddings(
        self, table_name: str, ids: Optional[List[str]] = None
    ) -> Dict[str, NDArray[np.float64]]:
        query = f"SELECT id, embedding FROM {table_name}"
        params = ()
        if ids is not None:
            query += " WHERE id = ANY(%s)"
            params = (ids,)

        with self.conn.cursor() as cur:
            cur.execute(query, params)
            results = cur.fetchall()
            return {row["id"]: np.array(row["embedding"]) for row in results}

    def count_embeddings(self, table_name: str) -> int:
        query = f"SELECT COUNT(*) FROM {table_name};"
        with self.conn.cursor() as cur:
            cur.execute(query)
            result = cur.fetchone()
            return result["count"] if result else 0

    def get_embedding_by_id(
        self, table_name: str, embedding_id: str
    ) -> Optional[NDArray[np.float64]]:
        query = f"SELECT embedding FROM {table_name} WHERE id = %s;"
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

        if dimension is None:
            dimension = len(self._to_list(embedding))

        embedding_id = str(uuid.uuid4())
        embedding_list = self._to_list(embedding)
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, embedding_list))
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
        if dimension is None:
            dimension = len(self._to_list(embeddings[0]))

        embedding_ids = [str(uuid.uuid4()) for _ in embeddings]
        formatted_embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in embeddings
        ]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_ids, formatted_embeddings))
        return embedding_ids

    def insert_embedding_by_id(
        self,
        table_name: str,
        embedding_id: str,
        embedding: EmbeddingInput,
        dimension: Optional[int] = None,
    ) -> None:
        if dimension is None:
            dimension = len(self._to_list(embedding))

        formatted_embedding = f"[{', '.join(map(str, self._to_list(embedding)))}]"
        query = f"INSERT INTO {table_name} (id, embedding) VALUES (%s, %s::vector);"
        with self.conn.cursor() as cur:
            cur.execute(query, (embedding_id, formatted_embedding))

    def insert_embeddings_by_ids(
        self,
        table_name: str,
        embedding_data: Dict[str, EmbeddingInput],
        dimension: Optional[int] = None,
    ) -> None:
        if not embedding_data:
            raise ValueError("Cannot insert empty embedding data")
        if dimension is None:
            dimension = len(self._to_list(next(iter(embedding_data.values()))))

        ids = list(embedding_data.keys())
        formatted_embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]"
            for v in embedding_data.values()
        ]
        query = f"INSERT INTO {table_name} (id, embedding) SELECT UNNEST(%s::text[]), UNNEST(%s::vector[]);"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, formatted_embeddings))

    def update_embedding_by_id(
        self, table_name: str, embedding_id: str, new_embedding: EmbeddingInput
    ) -> None:
        query = f"UPDATE {table_name} SET embedding = %s WHERE id = %s;"
        with self.conn.cursor() as cur:
            cur.execute(query, (self._to_list(new_embedding), embedding_id))

    def update_embedding_by_ids(
        self, table_name: str, updates: Dict[str, EmbeddingInput]
    ) -> None:
        ids = list(updates.keys())
        embeddings = [
            f"[{', '.join(map(str, self._to_list(v)))}]" for v in updates.values()
        ]
        query = f"UPDATE {table_name} SET embedding = data.embedding FROM (SELECT UNNEST(%s::text[]) AS id, UNNEST(%s::vector[]) AS embedding) AS data WHERE {table_name}.id = data.id;"
        with self.conn.cursor() as cur:
            cur.execute(query, (ids, embeddings))
