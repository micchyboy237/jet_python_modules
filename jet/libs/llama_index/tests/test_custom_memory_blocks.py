import pytest
import asyncio
from typing import List
from llama_index.core.memory import Memory
from jet.libs.llama_index.custom_memory_blocks import run_custom_memory
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


@pytest.fixture
def setup_memory():
    """Fixture to clean up memory state and vector store."""
    memory = Memory.from_defaults(
        session_id="custom_session", token_limit=40000)
    client = chromadb.EphemeralClient()
    vector_store = ChromaVectorStore(
        chroma_collection=client.create_collection("test_collection"))
    yield memory, vector_store
    # Clean up (no database to clear, in-memory SQLite and ephemeral Chroma)


class TestCustomMemoryBlocks:
    """Tests for custom memory blocks."""

    @pytest.mark.asyncio
    async def test_custom_memory_run(self, setup_memory):
        # Given: A question and vector store
        memory, vector_store = setup_memory
        question = "Who am I?"
        tools = []
        # Expect a string response (actual content depends on LLM)
        expected_response = str

        # When: Running the custom memory example
        result = await run_custom_memory(question, tools, vector_store)

        # Then: Verify the response is a string
        assert isinstance(
            result, expected_response), f"Expected {expected_response}, got {type(result)}"
