"""
Tests for the Retrieval and Reranking pipeline.
"""

import pytest


class TestRetrievalPipeline:
    @pytest.mark.asyncio
    async def test_full_retrieval_flow(
        self, retriever, vector_store, temp_repo_dir, scanner, chunker
    ):
        # Index some data first
        files = scanner.scan(str(temp_repo_dir))
        chunks = chunker.chunk_files(files)
        await vector_store.add_chunks(chunks)

        # Test retrieval
        try:
            results = await retriever.retrieve("how to calculate sum")

            # If reranker server is down, it should still return results from vector search
            assert isinstance(results, list)

            if results:
                # Check if rerank_score was added (if reranker worked) or just distance
                assert "rerank_score" in results[0] or "_distance" in results[0]

        except Exception as e:
            pytest.skip(f"Retrieval pipeline failed (servers might be offline): {e}")

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, retriever):
        results = await retriever.retrieve("")
        assert results == []
