import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage
from jet.libs.llama_index.mention_counter_memory import run_mention_counter, MentionCounter


@pytest.fixture
def setup_memory():
    """Fixture to clean up memory state."""
    memory = Memory.from_defaults(
        session_id="mention_session",
        token_limit=40000,
        memory_blocks=[MentionCounter(name="mention_counter", priority=0)],
    )
    yield memory
    # Clean up (no database to clear, in-memory SQLite)


class TestMentionCounterMemory:
    """Tests for MentionCounter memory block."""

    @pytest.mark.asyncio
    async def test_mention_counter(self, setup_memory):
        # Given: A question and a message mentioning Logan
        question = "How many times was Logan mentioned?"
        tools = []
        # Expect a string response (actual content depends on LLM)
        expected_response = str

        # When: Running the mention counter example
        result = await run_mention_counter(question, tools)

        # Then: Verify the response is a string
        assert isinstance(
            result, expected_response), f"Expected {expected_response}, got {type(result)}"

        # Additional check: Verify mention count
        chat_history = setup_memory.get()
        assert any(
            "Logan was mentioned 1 times" in msg.content for msg in chat_history), "Expected mention count not found"
