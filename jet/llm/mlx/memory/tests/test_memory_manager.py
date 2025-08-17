import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from autogen_core.memory import MemoryContent, MemoryMimeType, MemoryQueryResult, UpdateContextResult
from autogen_core.models import UserMessage, AssistantMessage
from autogen_core.model_context import ChatCompletionContext
from jet.llm.mlx.memory import MemoryManager
from jet.models.model_types import EmbedModelType, LLMModelType


@pytest.fixture
def mock_mem0_memory():
    """Fixture to mock Mem0Memory."""
    mem0_mock = AsyncMock()
    mem0_mock.add = AsyncMock()
    mem0_mock.query = AsyncMock()
    mem0_mock.clear = AsyncMock()
    mem0_mock.close = AsyncMock()
    return mem0_mock


@pytest.fixture
def mock_sentence_transformer():
    """Fixture to mock SentenceTransformerRegistry."""
    transformer_mock = MagicMock()
    transformer_mock.load_model.return_value = MagicMock()
    return transformer_mock


@pytest.fixture
async def memory_manager(mock_mem0_memory, mock_sentence_transformer, monkeypatch):
    """Fixture to create MemoryManager with mocked dependencies."""
    monkeypatch.setattr("jet.llm.mlx.memory.memory_manager.SentenceTransformerRegistry.load_model",
                        mock_sentence_transformer.load_model)
    manager = MemoryManager(
        user_id="test_user",
        limit=5,
        llm_model_path="qwen3-1.7b-4bit",
        embedder_model_path="all-MiniLM-L6-v2",
    )
    manager.memory = mock_mem0_memory
    return manager


@pytest.mark.asyncio
class TestMemoryManagerAdd:
    """Tests for MemoryManager.add method."""

    async def test_add_text_memory(self, memory_manager, mock_mem0_memory):
        """Given a text memory content, when adding to MemoryManager, then it should call Mem0Memory.add."""
        # Given
        content = MemoryContent(
            content="Meal recipe must be vegan",
            mime_type=MemoryMimeType.TEXT,
            metadata={"category": "preferences", "type": "dietary"}
        )
        expected_call = content

        # When
        await memory_manager.add(content)

        # Then
        mock_mem0_memory.add.assert_awaited_once_with(
            content, cancellation_token=None)


@pytest.mark.asyncio
class TestMemoryManagerQuery:
    """Tests for MemoryManager.query method."""

    async def test_query_with_string(self, memory_manager, mock_mem0_memory):
        """Given a string query, when querying MemoryManager, then it should return MemoryQueryResult."""
        # Given
        query = "What are my dietary preferences?"
        expected_results = [MemoryContent(
            content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT)]
        mock_mem0_memory.query.return_value = MemoryQueryResult(
            results=expected_results)

        # When
        result = await memory_manager.query(query)

        # Then
        mock_mem0_memory.query.assert_awaited_once_with(
            query, cancellation_token=None)
        assert isinstance(result, MemoryQueryResult)
        assert result.results == expected_results

    async def test_query_with_memory_content(self, memory_manager, mock_mem0_memory):
        """Given a MemoryContent query, when querying MemoryManager, then it should return MemoryQueryResult."""
        # Given
        query = MemoryContent(content="Vegan preferences",
                              mime_type=MemoryMimeType.TEXT)
        expected_results = [MemoryContent(
            content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT)]
        mock_mem0_memory.query.return_value = MemoryQueryResult(
            results=expected_results)

        # When
        result = await memory_manager.query(query)

        # Then
        mock_mem0_memory.query.assert_awaited_once_with(
            query, cancellation_token=None)
        assert isinstance(result, MemoryQueryResult)
        assert result.results == expected_results


@pytest.mark.asyncio
class TestMemoryManagerUpdateContext:
    """Tests for MemoryManager.update_context method."""

    async def test_update_context_with_user_message(self, memory_manager, mock_mem0_memory):
        """Given a context with a user message, when updating context, then it should query with the message content."""
        # Given
        context = MagicMock(spec=ChatCompletionContext)
        context.get_messages = AsyncMock(return_value=[
            UserMessage(
                content="What are my dietary preferences?", role="user")
        ])
        expected_query = "What are my dietary preferences?"
        expected_results = [MemoryContent(
            content="Meal recipe must be vegan", mime_type=MemoryMimeType.TEXT)]
        mock_mem0_memory.query.return_value = MemoryQueryResult(
            results=expected_results)

        # When
        result = await memory_manager.update_context(context)

        # Then
        mock_mem0_memory.query.assert_awaited_once_with(
            expected_query, cancellation_token=None)
        assert isinstance(result, UpdateContextResult)
        assert result.memories.results == expected_results

    async def test_update_context_with_empty_messages(self, memory_manager, mock_mem0_memory):
        """Given a context with no messages, when updating context, then it should query with empty string."""
        # Given
        context = MagicMock(spec=ChatCompletionContext)
        context.get_messages = AsyncMock(return_value=[])
        expected_query = ""
        expected_results = []
        mock_mem0_memory.query.return_value = MemoryQueryResult(
            results=expected_results)

        # When
        result = await memory_manager.update_context(context)

        # Then
        mock_mem0_memory.query.assert_awaited_once_with(
            expected_query, cancellation_token=None)
        assert isinstance(result, UpdateContextResult)
        assert result.memories.results == expected_results


@pytest.mark.asyncio
class TestMemoryManagerClear:
    """Tests for MemoryManager.clear method."""

    async def test_clear_memory(self, memory_manager, mock_mem0_memory):
        """Given a MemoryManager with memories, when clear is called, then it should call Mem0Memory.clear."""
        # When
        await memory_manager.clear()

        # Then
        mock_mem0_memory.clear.assert_awaited_once()

    async def test_clear_memory_unsupported(self, memory_manager, mock_mem0_memory, monkeypatch):
        """Given a Mem0Memory without clear, when clear is called, then it should raise NotImplementedError."""
        # Given
        del mock_mem0_memory.clear

        # When/Then
        with pytest.raises(NotImplementedError, match="Mem0Memory does not support clear operation"):
            await memory_manager.clear()


@pytest.mark.asyncio
class TestMemoryManagerClose:
    """Tests for MemoryManager.close method."""

    async def test_close_memory(self, memory_manager, mock_mem0_memory):
        """Given a MemoryManager, when close is called, then it should call Mem0Memory.close."""
        # When
        await memory_manager.close()

        # Then
        mock_mem0_memory.close.assert_awaited_once()

    async def test_close_memory_no_close_method(self, memory_manager, mock_mem0_memory):
        """Given a Mem0Memory without close, when close is called, then it should pass silently."""
        # Given
        del mock_mem0_memory.close

        # When
        await memory_manager.close()

        # Then
        # No exception should be raised, and no calls to a non-existent close method
