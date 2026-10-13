"""
Tests for the main Query Engine orchestration.
"""

import pytest


class TestQueryEngine:
    @pytest.mark.asyncio
    async def test_query_no_context(self, query_engine):
        """Test behavior when no relevant info is found."""

        # Mock retriever to return empty list
        async def mock_retrieve_empty(query, filters=None):
            return []

        # We need to monkeypatch the retriever inside the engine
        original_retriever = query_engine.retriever
        original_retrieve = original_retriever.retrieve

        try:
            original_retriever.retrieve = mock_retrieve_empty
            result = await query_engine.query("xyz_nonexistent_function_123")
            assert "couldn't find relevant information" in result["answer"].lower()
            assert result["sources"] == []
        finally:
            original_retriever.retrieve = original_retrieve

    @pytest.mark.asyncio
    async def test_query_with_context_mock(self, query_engine):
        """Test query generation by mocking the retrieval and LLM call."""

        # Mock retrieval to return fake context
        async def mock_retrieve(query, filters=None):
            return [
                {
                    "content": 'def hello(): return "world"',
                    "file_path": "test.py",
                    "start_line": 1,
                    "end_line": 1,
                    "rerank_score": 0.9,
                }
            ]

        # Mock LLM call
        async def mock_llm(question, context):
            return "The function returns the string world."

        # Apply mocks
        original_retriever = query_engine.retriever
        original_retrieve = original_retriever.retrieve
        original_llm = query_engine._call_llm

        try:
            original_retriever.retrieve = mock_retrieve
            query_engine._call_llm = mock_llm

            result = await query_engine.query("what does hello do?")

            assert "world" in result["answer"].lower()
            assert len(result["sources"]) == 1
            assert result["sources"][0]["file"] == "test.py"
        finally:
            original_retriever.retrieve = original_retrieve
            query_engine._call_llm = original_llm

    @pytest.mark.asyncio
    async def test_token_truncation(self, query_engine):
        """Ensure context doesn't exceed token limits."""
        long_content = "A" * 20000  # ~5000 tokens

        async def mock_retrieve(query, filters=None):
            return [
                {
                    "content": long_content,
                    "file_path": "big.py",
                    "start_line": 1,
                    "end_line": 1000,
                    "rerank_score": 0.9,
                }
            ]

        original_retriever = query_engine.retriever
        original_retrieve = original_retriever.retrieve

        try:
            original_retriever.retrieve = mock_retrieve

            # We expect the context builder to truncate this
            context_str, _ = query_engine._build_context(
                await mock_retrieve("test"), max_tokens=1000
            )

            assert len(context_str) < len(long_content)
        finally:
            original_retriever.retrieve = original_retrieve
