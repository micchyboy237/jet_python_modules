# jet_python_modules/jet/db/postgres/managers/search.py
from typing import Dict, List, Literal

import numpy as np
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

    def diverse_search(
        self,
        table_name: str,
        query_vector: EmbeddingInput,
        top_k: int = 5,
        num_candidates: int = 20,
        lambda_mult: float = 0.5,
        metric: Literal["cosine", "l2"] = "cosine",
    ) -> List[Dict]:
        """
        Perform Maximal Marginal Relevance (MMR) search.
        """
        if lambda_mult < 0 or lambda_mult > 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        vec = self._to_vector(query_vector)

        # Fix: Use sql.SQL for operators
        op = sql.SQL("<=>") if metric == "cosine" else sql.SQL("<->")

        with self.conn.cursor() as cur:
            # 1. Fetch initial candidates
            query = sql.SQL(
                "SELECT id, embedding FROM {} ORDER BY embedding {} %s LIMIT %s"
            ).format(sql.Identifier(table_name), op)

            cur.execute(query, (vec, num_candidates))
            candidates = cur.fetchall()

            if not candidates:
                return []

            # Convert to numpy for easier math
            # Fix: Explicitly convert Vector objects to lists before creating numpy arrays
            query_emb = np.array(vec.to_list() if hasattr(vec, "to_list") else vec)

            candidate_ids = [c["id"] for c in candidates]
            candidate_embs = np.array(
                [
                    np.array(
                        c["embedding"].to_list()
                        if hasattr(c["embedding"], "to_list")
                        else c["embedding"]
                    )
                    for c in candidates
                ]
            )

            # 2. MMR Algorithm
            selected_indices = []
            remaining_indices = list(range(len(candidates)))

            # Select the first item (most relevant)
            first_idx = remaining_indices.pop(0)
            selected_indices.append(first_idx)

            while len(selected_indices) < top_k and remaining_indices:
                best_idx = None
                best_score = -float("inf")

                for idx in remaining_indices:
                    # Relevance to query
                    if metric == "cosine":
                        sim_query = np.dot(candidate_embs[idx], query_emb) / (
                            np.linalg.norm(candidate_embs[idx])
                            * np.linalg.norm(query_emb)
                            + 1e-9
                        )
                    else:
                        sim_query = -np.linalg.norm(candidate_embs[idx] - query_emb)

                    # Redundancy (max similarity to already selected items)
                    max_sim_selected = -float("inf")
                    for sel_idx in selected_indices:
                        if metric == "cosine":
                            sim_sel = np.dot(
                                candidate_embs[idx], candidate_embs[sel_idx]
                            ) / (
                                np.linalg.norm(candidate_embs[idx])
                                * np.linalg.norm(candidate_embs[sel_idx])
                                + 1e-9
                            )
                        else:
                            sim_sel = -np.linalg.norm(
                                candidate_embs[idx] - candidate_embs[sel_idx]
                            )
                        max_sim_selected = max(max_sim_selected, sim_sel)

                    # MMR Score
                    mmr_score = (
                        lambda_mult * sim_query - (1 - lambda_mult) * max_sim_selected
                    )

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

            # 3. Fetch full details for selected items
            selected_ids = [candidate_ids[i] for i in selected_indices]
            if not selected_ids:
                return []

            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))

            id_placeholders = sql.SQL(", ").join(
                [sql.Placeholder()] * len(selected_ids)
            )
            detail_query = sql.SQL("SELECT {} FROM {} WHERE id IN ({})").format(
                select_cols, sql.Identifier(table_name), id_placeholders
            )

            cur.execute(detail_query, selected_ids)
            return [dict(row) for row in cur.fetchall()]

    def multi_query_search(
        self,
        table_name: str,
        query_texts: List[str],
        top_k: int = 5,
        num_candidates: int = 20,
        rrf_k: int = 60,
    ) -> List[Dict]:
        """
        Perform search across multiple related queries and merge results using RRF.
        """
        if not query_texts:
            return []

        all_rankings = []

        # 1. Fetch rankings for each query
        for text in query_texts:
            from jet.adapters.llama_cpp.config import EMBED_QUERY_PREFIX
            from jet.adapters.llama_cpp.embed_utils import embed

            vec = self._to_vector(embed(text, prefix=EMBED_QUERY_PREFIX))

            with self.conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT id, RANK() OVER (ORDER BY embedding <=> %s) as rank FROM {} LIMIT %s"
                    ).format(sql.Identifier(table_name)),
                    (vec, num_candidates),
                )
                # Fetchall returns a list of dicts because of dict_row factory
                all_rankings.append(cur.fetchall())

        # 2. Apply Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}

        for rankings in all_rankings:
            for row in rankings:
                rid = row["id"]
                rank = row["rank"]
                rrf_scores[rid] = rrf_scores.get(rid, 0) + (1.0 / (rrf_k + rank))

        # 3. Sort by combined RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        if not sorted_items:
            return []

        winner_ids = [item[0] for item in sorted_items]

        # 4. Fetch full details for the winners using a NEW cursor
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))

            id_placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(winner_ids))
            detail_query = sql.SQL("SELECT {} FROM {} WHERE id IN ({})").format(
                select_cols, sql.Identifier(table_name), id_placeholders
            )

            cur.execute(detail_query, winner_ids)
            results_map = {row["id"]: dict(row) for row in cur.fetchall()}

        # 5. Maintain the order from RRF sorting
        final_results = []
        for rid, score in sorted_items:
            if rid in results_map:
                entry = results_map[rid]
                entry["rrf_score"] = score
                final_results.append(entry)

        return final_results
