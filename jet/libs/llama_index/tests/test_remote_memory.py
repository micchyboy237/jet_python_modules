import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from jet.libs.llama_index.remote_memory import run_remote_memory


@pytest.fixture
def setup_memory():
    """Fixture to clean up memory state."""
    memory = Memory.from_defaults(
        session_id="remote_session",
        token_limit=40000,
        async_database_uri="postgresql+asyncpg://postgres:jethroestrada@localhost:5432/postgres",
    )
    yield memory
    # Clean up (no database to clear, assumes external management)


class TestRemoteMemory:
    """Tests for remote memory with PostgreSQL."""

    @pytest.mark.asyncio
    async def test_remote_memory_run(self, setup_memory):
        # Given: A question and empty tools list
        question = "What's stored in memory?"
        tools = []
        # Expect a string response (actual content depends on LLM)
        expected_response = str

        # When: Running the remote memory example
        result = await run_remote_memory(question, tools)

        # Then: Verify the response is a string
        assert isinstance(
            result, expected_response), f"Expected {expected_response}, got {type(result)}"
