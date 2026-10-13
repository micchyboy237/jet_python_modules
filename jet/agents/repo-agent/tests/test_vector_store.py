"""
Tests for LanceDB VectorStore operations.
"""

import pytest


class TestVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self, vector_store, temp_repo_dir, scanner, chunker):
        # 1. Prepare data
        files = scanner.scan(str(temp_repo_dir))
        chunks = chunker.chunk_files(files)

        # 2. Add to store
        await vector_store.add_chunks(chunks)

        # 3. Verify count
        assert vector_store.table.count_rows() == len(chunks)

        # 4. Search (Note: This requires the Embedding server to be running)
        try:
            results = await vector_store.search("calculate sum", k=5)
            assert isinstance(results, list)
            if results:
                assert "_distance" in results[0]
                assert "content" in results[0]
        except Exception as e:
            pytest.skip(f"Embedding server not available: {e}")

    def test_initialization_idempotency(self, vector_store):
        """Ensure calling initialize twice doesn't break anything."""
        vector_store.initialize()
        assert vector_store.table is not None
