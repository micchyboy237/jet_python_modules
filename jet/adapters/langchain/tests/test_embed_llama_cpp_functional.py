# tests/functional/adapters/langchain/test_embed_llama_cpp_functional.py
"""
Functional tests for EmbedLlamaCpp using real embedding server.

Prerequisites:
- Run llama.cpp embedding server:
  ```bash
  ./server -m models/embeddinggemma.Q4_K_M.gguf --port 8081 --embeddings
  ```
- Server must expose /v1/embeddings endpoint.

Validates:
- Batch embedding
- Query embedding
- Format conversion
- Async fallback
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from jet.adapters.langchain.embed_llama_cpp import EmbedLlamaCpp


@pytest.fixture(scope="module")
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def embed_model_list() -> EmbedLlamaCpp:
    """Embedding model returning list format."""
    return EmbedLlamaCpp(
        model="embeddinggemma",
        base_url="http://shawn-pc.local:8081/v1",
        batch_size=2,
        return_format="list",
        show_progress=False,
    )


@pytest.fixture(scope="module")
def embed_model_numpy() -> EmbedLlamaCpp:
    """Embedding model returning numpy format (default)."""
    return EmbedLlamaCpp(
        model="embeddinggemma",
        base_url="http://shawn-pc.local:8081/v1",
        batch_size=2,
        # return_format omitted – defaults to "numpy"
        show_progress=False,
    )


class TestEmbedLlamaCppFunctional:
    # Given: Real embedding server
    # When: Embedding multiple documents
    # Then: Returns correct number of vectors with expected dimension
    def test_embed_documents_list(self, embed_model_list: EmbedLlamaCpp) -> None:
        texts = ["hello world", "embedding test", "functional validation"]
        embeddings = embed_model_list.embed_documents(texts)

        assert len(embeddings) == len(texts)
        assert all(isinstance(vec, list) for vec in embeddings)
        assert all(isinstance(x, float) for vec in embeddings for x in vec)
        # Typical nomic dimension
        assert len(embeddings[0]) in {768, 512, 1024}

    # Given: return_format="numpy"
    # When: Embedding documents
    # Then: Internally uses numpy, converts to list
    def test_embed_documents_numpy(self, embed_model_numpy: EmbedLlamaCpp) -> None:
        texts = ["numpy", "array", "conversion"]
        embeddings = embed_model_numpy.embed_documents(texts)

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        assert all(isinstance(vec, list) for vec in embeddings)

    # Given: Single query
    # When: embed_query
    # Then: Returns 1D vector
    def test_embed_query(self, embed_model_list: EmbedLlamaCpp) -> None:
        embedding = embed_model_list.embed_query("single query")

        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
        assert len(embedding) > 100  # reasonable embedding size

    # Given: Async method
    # When: aembed_documents
    # Then: Falls back to sync, returns correct result
    @pytest.mark.asyncio
    async def test_aembed_documents(self, embed_model_list: EmbedLlamaCpp) -> None:
        texts = ["async", "embed"]
        embeddings = await embed_model_list.aembed_documents(texts)

        assert len(embeddings) == 2
        assert isinstance(embeddings[0], list)

    # Given: Async query
    # When: aembed_query
    # Then: Returns embedding
    @pytest.mark.asyncio
    async def test_aembed_query(self, embed_model_list: EmbedLlamaCpp) -> None:
        embedding = await embed_model_list.aembed_query("async query")

        assert isinstance(embedding, list)
        assert len(embedding) > 100

    # Given: Identical inputs
    # When: Embedding twice
    # Then: Results are nearly identical (within float tolerance)
    def test_embedding_determinism(self, embed_model_list: EmbedLlamaCpp) -> None:
        text = "deterministic embedding test"
        emb1 = embed_model_list.embed_query(text)
        emb2 = embed_model_list.embed_query(text)

        # Convert to numpy for comparison
        arr1 = np.array(emb1)
        arr2 = np.array(emb2)
        assert np.allclose(arr1, arr2, rtol=1e-6, atol=1e-8)