# jet_python_modules/jet/db/postgres/managers/search.py
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from pgvector import Vector
from psycopg import sql

from jet.db.postgres.managers.filter_builder import FilterBuilder
from jet.db.postgres.pg_types import (
    DiverseSearchResult,
    EmbeddingInput,
    HybridSearchResult,
    MultiQueryResult,
    SemanticSearchResult,
)


class SearchManager:
    """Handles complex search strategies like Hybrid, RRF, and MMR."""

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
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticSearchResult]:
        vec = self._to_vector(query_vector)
        operators = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}
        op = operators.get(metric, "<=>")

        where_clause, where_params = FilterBuilder.build(filters)

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = 'public' AND table_name = %s AND column_name != 'embedding';",
                (table_name,),
            )
            cols = [r["column_name"] for r in cur.fetchall()]
            select_cols = sql.SQL(", ").join(map(sql.Identifier, cols + ["id"]))

            query = sql.SQL(
                "SELECT {}, embedding {} %s AS distance FROM {}{} ORDER BY distance LIMIT %s"
            ).format(select_cols, sql.SQL(op), sql.Identifier(table_name), where_clause)

            final_params = [vec] + where_params + [top_k]
            cur.execute(query, final_params)

            results: List[SemanticSearchResult] = []
            for i, row in enumerate(cur.fetchall(), start=1):
                d = dict(row)
                d["rank"] = i
                results.append(d)
            return results

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
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        vec = self._to_vector(query_vector)
        where_clause, where_params = FilterBuilder.build(filters)

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
                    FROM {table}{where}
                ),
                keyword_search AS (
                    SELECT id, RANK() OVER (ORDER BY ts_rank(to_tsvector('english', {content_col}), websearch_to_tsquery('english', %s)) DESC) as k_rank
                    FROM {table}
                    WHERE to_tsvector('english', {content_col}) @@ websearch_to_tsquery('english', %s){where}
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
                where=where_clause,
            )

            final_params = (
                [vec]
                + where_params
                + [query_text, query_text]
                + where_params
                + [rrf_k, text_weight, rrf_k, vector_weight, top_k]
            )

            cur.execute(query, final_params)
            results = cur.fetchall()
            if not results:
                return []

            ids = [r["id"] for r in results]
            id_placeholders = sql.SQL(", ").join([sql.Placeholder()] * len(ids))
            detail_query = sql.SQL("SELECT {} FROM {} WHERE id IN ({})").format(
                select_cols, sql.Identifier(table_name), id_placeholders
            )
            cur.execute(detail_query, ids)
            detailed_rows = {r["id"]: dict(r) for r in cur.fetchall()}

            final_results: List[HybridSearchResult] = []
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
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DiverseSearchResult]:
        if lambda_mult < 0 or lambda_mult > 1:
            raise ValueError("lambda_mult must be between 0 and 1")

        vec = self._to_vector(query_vector)
        op = sql.SQL("<=>") if metric == "cosine" else sql.SQL("<->")
        where_clause, where_params = FilterBuilder.build(filters)

        with self.conn.cursor() as cur:
            query = sql.SQL(
                "SELECT id, embedding FROM {}{} ORDER BY embedding {} %s LIMIT %s"
            ).format(sql.Identifier(table_name), where_clause, op)

            cur.execute(query, [vec] + where_params + [num_candidates])
            candidates = cur.fetchall()

            if not candidates:
                return []

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

            selected_indices = []
            remaining_indices = list(range(len(candidates)))

            first_idx = remaining_indices.pop(0)
            selected_indices.append(first_idx)

            while len(selected_indices) < top_k and remaining_indices:
                best_idx = None
                best_score = -float("inf")

                for idx in remaining_indices:
                    if metric == "cosine":
                        sim_query = np.dot(candidate_embs[idx], query_emb) / (
                            np.linalg.norm(candidate_embs[idx])
                            * np.linalg.norm(query_emb)
                            + 1e-9
                        )
                    else:
                        sim_query = -np.linalg.norm(candidate_embs[idx] - query_emb)

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

                    mmr_score = (
                        lambda_mult * sim_query - (1 - lambda_mult) * max_sim_selected
                    )

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

                if best_idx is not None:
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)

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
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MultiQueryResult]:
        if not query_texts:
            return []

        where_clause, where_params = FilterBuilder.build(filters)
        all_rankings = []

        for text in query_texts:
            from jet.adapters.llama_cpp.config import EMBED_QUERY_PREFIX
            from jet.adapters.llama_cpp.embed_utils import embed

            vec = self._to_vector(embed(text, prefix=EMBED_QUERY_PREFIX))

            with self.conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT id, RANK() OVER (ORDER BY embedding <=> %s) as rank FROM {}{} LIMIT %s"
                    ).format(sql.Identifier(table_name), where_clause),
                    [vec] + where_params + [num_candidates],
                )
                all_rankings.append(cur.fetchall())

        rrf_scores: Dict[str, float] = {}

        for rankings in all_rankings:
            for row in rankings:
                rid = row["id"]
                rank = row["rank"]
                rrf_scores[rid] = rrf_scores.get(rid, 0) + (1.0 / (rrf_k + rank))

        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[
            :top_k
        ]

        if not sorted_items:
            return []

        winner_ids = [item[0] for item in sorted_items]

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

        final_results: List[MultiQueryResult] = []
        for rid, score in sorted_items:
            if rid in results_map:
                entry = results_map[rid]
                entry["rrf_score"] = score
                final_results.append(entry)

        return final_results
