# jet_python_modules/jet/adapters/llama_cpp/vector_search.py
import os
from collections.abc import Iterator
from typing import Literal

from jet.adapters.llama_cpp.embeddings import LlamacppEmbedding
from jet.adapters.llama_cpp.types import (
    LLAMACPP_EMBED_KEYS,
    EmbeddingInputType,
    IDType,
    MetadataType,
    SearchResultType,
)
from jet.adapters.llama_cpp.utils import cosine_similarity
from jet.logger import CustomLogger


class VectorSearch:
    """
    Handles vector similarity search using a provided embedding function.

    Designed to work with any embedding provider that can embed strings → vectors.
    """

    def __init__(
        self,
        model: LLAMACPP_EMBED_KEYS | LlamacppEmbedding = "nomic-embed-text",
        *,
        normalize: bool = True,  # future extension point
        score_type: Literal["cosine"] = "cosine",
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        base_url: str | None = os.getenv("LLAMA_CPP_EMBED_URL"),
        use_cache: bool = False,
        verbose: bool = True,
        logger: CustomLogger | None = None,
    ):
        if isinstance(model, str):
            model = LlamacppEmbedding(
                model,
                base_url=base_url,
                use_cache=use_cache,
                verbose=verbose,
                logger=logger,
            )

        self.model: LlamacppEmbedding = model
        self.normalize = normalize
        self.score_type = score_type
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def _apply_prefix(
        self,
        texts: list[str],
        *,
        input_type: "EmbeddingInputType" = "default",
    ) -> list[str]:
        if input_type == "query" and self.query_prefix:
            return [f"{self.query_prefix}{t}" for t in texts]

        if input_type == "document" and self.document_prefix:
            return [f"{self.document_prefix}{t}" for t in texts]

        return texts

    def search(
        self,
        query: str,
        documents: list[str],
        *,
        ids: list[IDType | None] | None = None,
        metadatas: list[MetadataType | None] | None = None,
        top_k: int | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> list[SearchResultType]:
        """
        Perform semantic search: embed query + all documents in one pass,
        compute cosine similarities, sort by descending score.

        Optional per-document ids and metadatas are preserved in results
        when provided (must be same length as documents).
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not documents:
            return []

        n_docs = len(documents)

        if ids is not None and len(ids) != n_docs:
            raise ValueError(
                f"'ids' must be None or have length {n_docs}, got {len(ids)}"
            )
        if metadatas is not None and len(metadatas) != n_docs:
            raise ValueError(
                f"'metadatas' must be None or have length {n_docs}, got {len(metadatas)}"
            )

        # Combine query + documents → single embedding call
        all_texts = [query] + documents

        all_embs_list = self.model.get_embeddings(
            all_texts,
            return_format="list",
            batch_size=batch_size,
            show_progress=show_progress,
            use_cache=use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing,
        )

        # First embedding belongs to the query
        query_emb = all_embs_list[0]
        doc_embs = all_embs_list[1:]

        results: list[SearchResultType] = []
        for i, (text, emb) in enumerate(zip(documents, doc_embs)):
            score = cosine_similarity(query_emb, emb)
            item: SearchResultType = {
                "index": i,
                "text": text,
                "score": score,
            }
            if ids is not None:
                item["id"] = ids[i]
            if metadatas is not None:
                item["metadata"] = metadatas[i]
            results.append(item)

        results.sort(key=lambda x: x["score"], reverse=True)
        if top_k is not None:
            results = results[:top_k]

        # Assign ranks (1-based)
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        return results

    def search_stream(
        self,
        query: str,
        documents: list[str],
        *,
        ids: list[IDType | None] | None = None,
        metadatas: list[MetadataType | None] | None = None,
        top_k: int | None = None,
        batch_size: int = 32,
        show_progress: bool = True,
        use_cache: bool | None = None,
        use_dynamic_batch_sizing: bool | None = None,
    ) -> Iterator[SearchResultType]:
        """
        Streaming version — yields one SearchResultType per document
        as soon as its embedding is computed.

        Note: top_k is currently NOT respected in streaming mode
        (all documents are yielded). Post-filtering must be done by consumer.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        if not documents:
            return

        n_docs = len(documents)

        if ids is not None and len(ids) != n_docs:
            raise ValueError(
                f"'ids' must be None or have length {n_docs}, got {len(ids)}"
            )
        if metadatas is not None and len(metadatas) != n_docs:
            raise ValueError(
                f"'metadatas' must be None or have length {n_docs}, got {len(metadatas)}"
            )

        formatted_queries = (
            self._apply_prefix([query], input_type="query")
            if self.query_prefix
            else [query]
        )
        formatted_docs = (
            self._apply_prefix(documents, input_type="document")
            if self.document_prefix
            else documents
        )

        embeddings_stream = self.model.get_embeddings_stream(
            inputs=formatted_queries + formatted_docs,
            return_format="numpy",
            batch_size=batch_size,
            show_progress=show_progress,
            use_cache=use_cache,
            use_dynamic_batch_sizing=use_dynamic_batch_sizing,
        )

        doc_counter = 0
        query_embedding = None

        for batch_embeddings in embeddings_stream:
            if query_embedding is None:
                # First batch contains query + possibly some documents
                query_embedding = batch_embeddings[0]
                doc_embeddings = batch_embeddings[1:]
            else:
                doc_embeddings = batch_embeddings

            for emb in doc_embeddings:
                if doc_counter >= len(documents):
                    # safety — should not happen
                    break

                score = cosine_similarity(query_embedding, emb)

                result: SearchResultType = {
                    "index": doc_counter,
                    "text": documents[doc_counter],
                    "score": score,
                }

                if ids is not None:
                    result["id"] = ids[doc_counter]
                if metadatas is not None:
                    result["metadata"] = metadatas[doc_counter]

                yield result

                doc_counter += 1

            if doc_counter >= len(documents):
                break
