# jet_python_modules/jet/db/postgres/pgvector.py
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .config import (
    DEFAULT_DB,
    DEFAULT_HOST,
    DEFAULT_PASSWORD,
    DEFAULT_PORT,
    DEFAULT_USER,
)
from .managers.connection import ConnectionManager
from .managers.metadata import MetadataManager
from .managers.query import QueryExecutor
from .managers.schema import SchemaManager
from .managers.search import SearchManager
from .managers.vector import VectorEngine
from .pg_types import (
    DatabaseMetadata,
    DiverseSearchResult,
    Embedding,
    EmbeddingInput,
    HybridSearchResult,
    MultiQueryResult,
    SearchResult,
    SemanticSearchResult,
    TableMetadata,
    TableRow,
)


class PgVectorClient:
    def __init__(
        self,
        dbname: str = DEFAULT_DB,
        user: str = DEFAULT_USER,
        password: str = DEFAULT_PASSWORD,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        overwrite_db: bool = False,
    ):
        self.connection = ConnectionManager(
            dbname, user, password, host, port, overwrite_db
        )
        self.schema = SchemaManager(self.connection.conn)
        self.query = QueryExecutor(self.connection.conn)
        self.metadata = MetadataManager(self.connection.conn)
        self.vector = VectorEngine(self.connection.conn)
        self.search_manager = SearchManager(self.connection.conn)
        self.conn = self.connection.conn

    def create_vector_table(
        self,
        table_name: str,
        dimension: int,
        additional_columns: Optional[Dict[str, str]] = None,
        include_timestamps: bool = True,
    ):
        cols = {"id": "TEXT PRIMARY KEY", "embedding": f"vector({dimension})"}
        if additional_columns:
            cols.update(additional_columns)
        self.schema.create_table(table_name, cols, include_timestamps)

    def create_table(self, table_name: str, dimension: int):
        self.create_vector_table(table_name, dimension)

    def create_index(
        self,
        table_name: str,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
        method: Literal["hnsw", "ivfflat"] = "hnsw",
        index_name: Optional[str] = None,
    ):
        self.vector.create_index(table_name, index_name, metric, method)

    def create_fts_index(self, table_name: str, column_name: str = "content"):
        self.schema.create_fts_index(table_name, column_name)

    def create_row(
        self, table_name: str, row_data: Dict[str, Any], dimension: Optional[int] = None
    ) -> TableRow:
        if "embedding" in row_data and dimension is None:
            emb = row_data["embedding"]
            dimension = len(emb) if not isinstance(emb, np.ndarray) else emb.shape[0]
        self.schema.ensure_columns_exist(table_name, row_data)
        casts = {"embedding": "::vector"} if "embedding" in row_data else None
        return self.query.insert_row(table_name, row_data, type_casts=casts)

    def create_rows(
        self,
        table_name: str,
        rows_data: List[Dict[str, Any]],
        dimension: Optional[int] = None,
    ) -> List[TableRow]:
        return [self.create_row(table_name, r, dimension) for r in rows_data]

    def update_row(
        self,
        table_name: str,
        row_id: str,
        row_data: Dict[str, Any],
        dimension: Optional[int] = None,
    ) -> TableRow:
        casts = {"embedding": "::vector"} if "embedding" in row_data else None
        return self.query.update_row(table_name, row_id, row_data, type_casts=casts)

    def update_rows(
        self,
        table_name: str,
        rows_data: List[Dict[str, Any]],
        dimension: Optional[int] = None,
    ) -> List[TableRow]:
        return [self.update_row(table_name, r["id"], r, dimension) for r in rows_data]

    def create_or_update_row(
        self, table_name: str, row_data: Dict[str, Any], dimension: Optional[int] = None
    ) -> TableRow:
        self.schema.ensure_columns_exist(table_name, row_data)
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM {} WHERE id = %s".format(table_name), (row_data["id"],)
            )
            if cur.fetchone():
                return self.update_row(table_name, row_data["id"], row_data, dimension)
            else:
                return self.create_row(table_name, row_data, dimension)

    def create_or_update_rows(
        self,
        table_name: str,
        rows_data: List[Dict[str, Any]],
        dimension: Optional[int] = None,
    ) -> List[TableRow]:
        return [self.create_or_update_row(table_name, r, dimension) for r in rows_data]

    def bulk_ingest(
        self,
        table_name: str,
        data: List[Dict[str, Any]],
        dimension: Optional[int] = None,
        batch_size: int = 1000,
    ) -> int:
        return self.vector.bulk_ingest(table_name, data, dimension, batch_size)

    def get_row(self, table_name: str, row_id: str) -> Optional[TableRow]:
        rows = self.query.get_rows(table_name, where_conditions={"id": row_id})
        return rows[0] if rows else None

    def get_rows(
        self,
        table_name: str,
        ids: Optional[List[str]] = None,
        id_column: str = "id",
        where_conditions: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        order_by: Optional[Tuple[str, Literal["ASC", "DESC"]]] = None,
        group_by: Optional[List[str]] = None,
        aggregates: Optional[
            Dict[str, Tuple[str, Literal["COUNT", "SUM", "AVG", "MIN", "MAX"]]]
        ] = None,
    ) -> List[TableRow]:
        return self.query.get_rows(
            table_name,
            ids,
            id_column,
            where_conditions,
            limit,
            order_by,
            group_by,
            aggregates,
        )

    def search(
        self,
        table_name: str,
        query_embedding: EmbeddingInput,
        top_k: int = 5,
        threshold: Optional[float] = None,
        metric: Literal["cosine", "l2", "ip"] = "cosine",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SemanticSearchResult]:
        results = self.search_manager.semantic_search(
            table_name, query_embedding, top_k, metric, filters
        )
        from jet.db.postgres.scoring import calculate_vector_scores

        final_results: List[SemanticSearchResult] = []
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
            for k, v in res.items():
                if k not in ["id", "distance"]:
                    entry[k] = v
            final_results.append(entry)
        return final_results

    def hybrid_search(
        self,
        table_name: str,
        query_text: str,
        query_embedding: EmbeddingInput,
        top_k: int = 5,
        rrf_k: int = 60,
        text_weight: float = 1.0,
        vector_weight: float = 1.0,
        content_column: str = "content",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[HybridSearchResult]:
        return self.search_manager.hybrid_search(
            table_name,
            query_text,
            query_embedding,
            top_k,
            rrf_k,
            text_weight,
            vector_weight,
            content_column,
            filters,
        )

    def diverse_search(
        self,
        table_name: str,
        query_embedding: EmbeddingInput,
        top_k: int = 5,
        num_candidates: int = 20,
        lambda_mult: float = 0.5,
        metric: Literal["cosine", "l2"] = "cosine",
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[DiverseSearchResult]:
        return self.search_manager.diverse_search(
            table_name,
            query_embedding,
            top_k,
            num_candidates,
            lambda_mult,
            metric,
            filters,
        )

    def multi_query_search(
        self,
        table_name: str,
        query_texts: List[str],
        top_k: int = 5,
        num_candidates: int = 20,
        rrf_k: int = 60,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MultiQueryResult]:
        return self.search_manager.multi_query_search(
            table_name, query_texts, top_k, num_candidates, rrf_k, filters
        )

    def get_embeddings(
        self, table_name: str, ids: List[str] = None
    ) -> Dict[str, Embedding]:
        return self.vector.get_embeddings(table_name, ids)

    def count_embeddings(self, table_name: str) -> int:
        return self.vector.count_embeddings(table_name)

    def get_embedding_by_id(
        self, table_name: str, embedding_id: str
    ) -> Optional[NDArray[np.float64]]:
        return self.vector.get_embedding_by_id(table_name, embedding_id)

    def insert_embedding(
        self,
        table_name: str,
        embedding: EmbeddingInput,
        dimension: Optional[int] = None,
    ) -> str:
        return self.vector.insert_embedding(table_name, embedding, dimension)

    def insert_embeddings(
        self,
        table_name: str,
        embeddings: List[EmbeddingInput],
        dimension: Optional[int] = None,
    ) -> List[str]:
        return self.vector.insert_embeddings(table_name, embeddings, dimension)

    def insert_embedding_by_id(
        self,
        table_name: str,
        embedding_id: str,
        embedding: EmbeddingInput,
        dimension: Optional[int] = None,
    ):
        self.vector.insert_embedding_by_id(
            table_name, embedding_id, embedding, dimension
        )

    def insert_embeddings_by_ids(
        self,
        table_name: str,
        embedding_data: Dict[str, EmbeddingInput],
        dimension: Optional[int] = None,
    ):
        self.vector.insert_embeddings_by_ids(table_name, embedding_data, dimension)

    def update_embedding_by_id(
        self, table_name: str, embedding_id: str, new_embedding: EmbeddingInput
    ):
        self.vector.update_embedding_by_id(table_name, embedding_id, new_embedding)

    def update_embedding_by_ids(
        self, table_name: str, updates: Dict[str, EmbeddingInput]
    ):
        self.vector.update_embedding_by_ids(table_name, updates)

    def create_enum_type(self, type_name: str, values: List[str]):
        self.schema.create_enum_type(type_name, values)

    def drop_enum_type(self, type_name: str, cascade: bool = True):
        self.schema.drop_enum_type(type_name, cascade)

    def get_database_metadata(self) -> DatabaseMetadata:
        return self.metadata.get_database_metadata()

    def get_all_tables(self) -> List[str]:
        return self.metadata.get_all_tables()

    def get_table_metadata(self, table_name: str) -> TableMetadata:
        return self.metadata.get_table_metadata(table_name)

    def get_database_summary(
        self,
    ) -> Dict[str, Union[DatabaseMetadata, List[TableMetadata]]]:
        return self.metadata.get_database_summary()

    def execute_raw_query(
        self, query: str, params: tuple | list = (), fetch_all: bool = True
    ):
        return self.query.execute_raw_query(query, params, fetch_all)

    def drop_all_rows(self, table_name: Optional[str] = None):
        self.query.drop_all_rows(table_name)

    def delete_all_tables(self):
        self.schema.delete_all_tables()

    def generate_unique_hash(self) -> str:
        return self.query.generate_unique_hash()

    def delete_db(self, confirm: bool = False):
        if confirm:
            self.connection.delete_db()

    def close(self):
        self.connection.close()

    def begin_transaction(self) -> None:
        self.connection.begin_transaction()

    def commit(self) -> None:
        self.connection.commit()

    def rollback(self) -> None:
        self.connection.rollback()

    def __enter__(self):
        return self.connection.__enter__()

    def __exit__(self, *args):
        return self.connection.__exit__(*args)


__all__ = ["PgVectorClient", "SearchResult"]
