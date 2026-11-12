from typing import Dict, List, Literal

from pgvector import Vector
from psycopg import sql

from jet.db.postgres.pg_types import EmbeddingInput


class SearchManager:
    """Handles complex search strategies like Hybrid and RRF."""

    def __init__(self, conn):
        self.conn = conn

    def _to_vector(self, embedding: EmbeddingInput) -> Vector:
        if isinstance(embedding, list):
            return Vector(embedding)
        return embedding

    def hybrid_search(
        self,
        table_name: str,
        query_text: str,
        query_vector: EmbeddingInput,
        top_k: int = 5,
        rrf_k: int = 60,
        text_weight: float = 1.0,
        vector_weight: float = 1.0,
        content_column: str = "content",
    ) -> List[Dict]:
        vec = self._to_vector(query_vector)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))

            query = sql.SQL("""
                WITH vector_search AS (
                    SELECT id, RANK() OVER (ORDER BY embedding <=> %s) as v_rank
                    FROM {table}
                ),
                keyword_search AS (
                    SELECT id, RANK() OVER (ORDER BY ts_rank(to_tsvector('english', {content_col}), websearch_to_tsquery('english', %s)) DESC) as k_rank
                    FROM {table}
                    WHERE to_tsvector('english', {content_col}) @@ websearch_to_tsquery('english', %s)
                )
                SELECT 
                    v.id, 
                    COALESCE(1.0 / (%s + k.k_rank), 0.0) * %s + 
                    COALESCE(1.0 / (%s + v.v_rank), 0.0) * %s AS hybrid_score
                FROM vector_search v
                FULL OUTER JOIN keyword_search k ON v.id = k.id
                ORDER BY hybrid_score DESC
                LIMIT %s
            """).format(
                table=sql.Identifier(table_name),
                content_col=sql.Identifier(content_column),
            )

            cur.execute(
                query,
                (
                    vec,
                    query_text,
                    query_text,
                    rrf_k,
                    text_weight,
                    rrf_k,
                    vector_weight,
                    top_k,
                ),
            )

            results = cur.fetchall()

            if not results:
                return []

            ids = [r["id"] for r in results]
            id_placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(ids))

            detail_query = sql.SQL("SELECT {} FROM {} WHERE id IN ({})").format(
                select_cols, sql.Identifier(table_name), id_placeholders
            )

            cur.execute(detail_query, ids)
            detailed_rows = {r["id"]: r for r in cur.fetchall()}

            final_results = []
            for res in results:
                rid = res["id"]
                if rid in detailed_rows:
                    entry = detailed_rows[rid]
                    entry["hybrid_score"] = res["hybrid_score"]
                    final_results.append(entry)

            return final_results

    def semantic_search(
        self,
        table_name: str,
        query_vector: EmbeddingInput,
        top_k: int = 5,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
    ) -> List[Dict]:
        vec = self._to_vector(query_vector)

        operators = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
        op = operators.get(metric, "<=>")

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))

            query = sql.SQL(
                "SELECT {}, embedding {} %s AS distance FROM {} ORDER BY distance LIMIT %s"
            ).format(select_cols, sql.SQL(op), sql.Identifier(table_name))

            cur.execute(query, (vec, top_k))
            return [dict(row) for row in cur.fetchall()]
