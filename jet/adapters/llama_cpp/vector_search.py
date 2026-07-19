# jet_python_modules/jet/adapters/llama_cpp/vector_search.py
import os
from collections.abc import Iterator
from typing import Literal

from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.adapters.llama_cpp.types import (
    LLAMACPP_EMBED_KEYS,
    EmbeddingInputType,
    IDType,
    MetadataType,
    SearchResultType,
)
from jet.logger import CustomLogger


class VectorSearch:
    """
    Handles vector similarity search using embed_utils.embed() for embeddings
    and scoring_utils.cosine_similarity() for scoring.

    Note: base_url / use_cache / use_dynamic_batch_sizing are accepted for
    backward compatibility with older call sites but are NOT functional —
    embed_utils.embed() has no caching layer, no dynamic batch sizing, and
    its OpenAI client base_url is fixed at import time from the
    LLAMA_CPP_EMBED_URL env var. A warning is logged if these are used.
    """

    def __init__(
        self,
        model: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL"),
        *,
        normalize: bool = True,
        score_type: Literal["cosine"] = "cosine",
        query_prefix: str | None = None,
        document_prefix: str | None = None,
        max_workers: int = 6,
        base_url: str | None = None,  # deprecated, kept for compatibility
        use_cache: bool = False,  # deprecated, kept for compatibility
        verbose: bool = True,
        logger: CustomLogger | None = None,
    ):
        self.model = model
        self.normalize = normalize
        self.score_type = score_type
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.max_workers = max_workers
        self.verbose = verbose
        self.logger = logger or CustomLogger(__name__)

        if base_url is not None:
            self.logger.warning(
                "VectorSearch: 'base_url' has no effect — embed_utils binds its "
                "client to LLAMA_CPP_EMBED_URL at import time. "
                "Set that env var before import instead."
            )
        if use_cache:
            self.logger.warning(
                "VectorSearch: 'use_cache=True' is ignored — embed_utils.embed() "
                "has no caching layer."
            )
        if self.verbose:
            self.logger.info(
                f"VectorSearch initialized (model={self.model!r}, "
                f"score_type={self.score_type!r}, normalize={self.normalize})"
            )

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

    def _warn_unsupported_kwargs(
        self, use_cache: bool | None, use_dynamic_batch_sizing: bool | None
    ) -> None:
        if use_cache is not None:
            self.logger.warning(
                "VectorSearch: 'use_cache' is ignored — not supported by embed_utils."
            )
        if use_dynamic_batch_sizing is not None:
            self.logger.warning(
                "VectorSearch: 'use_dynamic_batch_sizing' is ignored — "
                "not supported by embed_utils."
            )

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
        self._warn_unsupported_kwargs(use_cache, use_dynamic_batch_sizing)

        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not documents:
            self.logger.info("search(): no documents provided, returning []")
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

        formatted_query = self._apply_prefix([query], input_type="query")[0]
        formatted_docs = self._apply_prefix(documents, input_type="document")

        self.logger.info(
            f"search(): embedding 1 query + {n_docs} documents (model={self.model})"
        )
        all_texts = [formatted_query] + formatted_docs
        all_embeddings = embed(
            all_texts,
            model=self.model,
            return_format="list",
            batch_size=batch_size,
            show_progress=show_progress,
            max_workers=self.max_workers,
        )
        query_emb = all_embeddings[0]
        doc_embs = all_embeddings[1:]

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
        for rank, result in enumerate(results, start=1):
            result["rank"] = rank

        self.logger.info(f"search(): complete, {len(results)} results returned")
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
        Streaming version — yields one SearchResultType per document as soon
        as its batch's embedding is computed.

        Implementation note: embed_utils has no native streaming API, so this
        re-embeds the query once, then embeds documents in `batch_size` chunks
        (using embed_utils.embed()'s own internal parallelism per chunk) and
        yields results chunk-by-chunk, preserving original document order.

        Note: top_k is currently NOT respected in streaming mode
        (all documents are yielded). Post-filtering must be done by consumer.
        """
        self._warn_unsupported_kwargs(use_cache, use_dynamic_batch_sizing)

        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not documents:
            self.logger.info("search_stream(): no documents provided, nothing to yield")
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

        formatted_query = self._apply_prefix([query], input_type="query")[0]
        formatted_docs = self._apply_prefix(documents, input_type="document")

        self.logger.info(f"search_stream(): embedding query (model={self.model})")
        query_emb = embed(formatted_query, model=self.model, return_format="numpy")

        doc_counter = 0
        for start in range(0, n_docs, batch_size):
            batch_docs = formatted_docs[start : start + batch_size]
            self.logger.info(
                f"search_stream(): embedding batch [{start}:{start + len(batch_docs)}] "
                f"of {n_docs}"
            )
            batch_embs = embed(
                batch_docs,
                model=self.model,
                return_format="numpy",
                max_workers=self.max_workers,
                show_progress=show_progress,
                batch_size=batch_size,
            )
            for emb in batch_embs:
                score = cosine_similarity(query_emb, emb)
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

        self.logger.info(
            f"search_stream(): complete, {doc_counter} documents processed"
        )


if __name__ == "__main__":
    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("VectorSearch demo")
    print("=" * 60)

    searcher = VectorSearch()

    print("\n[1] search() — batch mode")
    results = searcher.search(query, docs, top_k=3)
    print(f"Query: {query}\n")
    for r in results:
        print(f"#{r['rank']}  {r['score']:.4f}  {r['text']}")

    print("\n[2] search_stream() — streaming mode")
    for r in searcher.search_stream(query, docs, batch_size=2):
        print(f"index={r['index']}  {r['score']:.4f}  {r['text']}")
