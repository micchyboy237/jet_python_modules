import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage
from jet.libs.llama_index.manual_memory_management import run_manual_memory


@pytest.fixture
def setup_memory():
    """Fixture to clean up memory state."""
    memory = Memory.from_defaults(
        session_id="manual_session", token_limit=40000)
    yield memory
    # Clean up (no database to clear, in-memory SQLite)


class TestManualMemory:
    """Tests for manual memory management."""

    @pytest.mark.asyncio
    async def test_manual_memory_management(self, setup_memory):
        # Given: A question and predefined messages
        question = "What's next?"
        tools = []
        expected_messages = [
            ChatMessage(role="user", content="Hello, world!"),
            ChatMessage(role="assistant", content="Hello, world to you too!"),
        ]

        # When: Running the manual memory example
        result = await run_manual_memory(question, tools)

        # Then: Verify the chat history matches the expected messages
        assert len(result) == len(
            expected_messages), f"Expected {len(expected_messages)} messages, got {len(result)}"
        for res, exp in zip(result, expected_messages):
            assert res.role == exp.role, f"Expected role {exp.role}, got {res.role}"
            assert res.content == exp.content, f"Expected content {exp.content}, got {res.content}"
