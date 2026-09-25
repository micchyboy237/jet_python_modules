from typing import Dict, List, Optional

from psycopg import errors, sql

from jet.db.postgres.scoring import calculate_vector_scores


class VectorEngine:
    """Handles pgvector specific operations."""

    def __init__(self, conn):
        self.conn = conn
        self._initialize_extension()

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

        # Get non-embedding columns
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

        with self.conn.cursor() as cur:
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
                    entry[c] = res[c]
            final_results.append(entry)

        return final_results
