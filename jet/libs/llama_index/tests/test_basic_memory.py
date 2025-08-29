import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from jet.libs.llama_index.basic_memory import run_basic_memory


@pytest.fixture
def setup_memory():
    """Fixture to clean up memory state."""
    memory = Memory.from_defaults(
        session_id="basic_session", token_limit=40000)
    yield memory
    # Clean up (no database to clear, in-memory SQLite)


class TestBasicMemory:
    """Tests for basic memory configuration."""

    @pytest.mark.asyncio
    async def test_basic_memory_run(self, setup_memory):
        # Given: A simple question and empty tools list
        question = "What's the weather like today?"
        tools = []
        # Expect a string response (actual content depends on LLM)
        expected_response = str

        # When: Running the basic memory example
        result = await run_basic_memory(question, tools)

        # Then: Verify the response is a string
        assert isinstance(
            result, expected_response), f"Expected {expected_response}, got {type(result)}"
