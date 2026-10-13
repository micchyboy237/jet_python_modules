"""
Tests for the Intent Classification Router.
"""

from unittest.mock import patch

import pytest


class TestIntentRouter:
    @pytest.mark.asyncio
    async def test_classify_factual(self, intent_router):
        """Test classification of a factual query."""

        async def mock_achat(*args, **kwargs):
            return type("obj", (object,), {"content": "FACTUAL"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            intent = await intent_router.classify("What does calculate_sum do?")
            assert intent == "FACTUAL"

    @pytest.mark.asyncio
    async def test_classify_summary(self, intent_router):
        """Test classification of a summary query."""

        async def mock_achat(*args, **kwargs):
            return type("obj", (object,), {"content": "SUMMARY"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            intent = await intent_router.classify("Summarize this repo's features")
            assert intent == "SUMMARY"

    @pytest.mark.asyncio
    async def test_classify_exploratory(self, intent_router):
        """Test classification of an exploratory query."""

        async def mock_achat(*args, **kwargs):
            return type("obj", (object,), {"content": "EXPLORATORY"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            intent = await intent_router.classify("List all API endpoints")
            assert intent == "EXPLORATORY"

    @pytest.mark.asyncio
    async def test_classify_debugging(self, intent_router):
        """Test classification of a debugging query."""

        async def mock_achat(*args, **kwargs):
            return type("obj", (object,), {"content": "DEBUGGING"})

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            intent = await intent_router.classify(
                "Why is this function throwing a KeyError?"
            )
            assert intent == "DEBUGGING"

    @pytest.mark.asyncio
    async def test_classify_fallback(self, intent_router):
        """Test fallback to FACTUAL if classification fails."""

        async def mock_achat(*args, **kwargs):
            raise Exception("LLM Error")

        with patch(
            "jet.adapters.llama_cpp.llm_utils_observed.achat", side_effect=mock_achat
        ):
            intent = await intent_router.classify("Test query")
            assert intent == "FACTUAL"
