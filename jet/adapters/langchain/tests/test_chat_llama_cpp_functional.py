# tests/functional/adapters/langchain/test_chat_llama_cpp_functional.py
"""
Functional tests for ChatLlamaCpp using a real llama.cpp server.

Prerequisites:
- Run a llama.cpp server with OpenAI-compatible API:
  ```bash
  ./server -m models/qwen3-instruct-2507-q4_0.gguf --port 8080 --host 0.0.0.0
  ```
- Ensure model supports chat and streaming.

These tests validate end-to-end behavior including:
- Message conversion
- Streaming
- Async support
- Stop sequences
"""

from __future__ import annotations

import asyncio
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp


@pytest.fixture(scope="module")
def event_loop() -> asyncio.AbstractEventLoop:
    """Create event loop for module-scoped async fixtures."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def chat_model() -> ChatLlamaCpp:
    """
    Real ChatLlamaCpp instance pointing to local server.
    Adjust base_url if your server runs on different port.
    """
    return ChatLlamaCpp(
        model="qwen3-instruct-2507:4b",  # must match model loaded in server
        base_url="http://shawn-pc.local:8080/v1",
        api_key="sk-1234",
        temperature=0.3,
        max_tokens=128,
    )


class TestChatLlamaCppFunctionalSync:
    # Given: Real server with chat model
    # When: Sending a simple user message
    # Then: Receives coherent AI response
    def test_chat_single_message(self, chat_model: ChatLlamaCpp) -> None:
        messages: List[BaseMessage] = [HumanMessage(content="What is the capital of France?")]
        result = chat_model.invoke(messages)

        assert isinstance(result, AIMessage)
        content = result.content.lower()
        assert "paris" in content

    # Given: System + user messages
    # When: Using system prompt to guide tone
    # Then: Response follows instruction
    def test_chat_with_system_prompt(self, chat_model: ChatLlamaCpp) -> None:
        messages = [
            SystemMessage(content="You are a concise assistant. Answer in one word."),
            HumanMessage(content="What color is the sky?"),
        ]
        result = chat_model.invoke(messages)

        assert isinstance(result, AIMessage)
        content = result.content.strip().lower()
        assert content in ["blue", "clear", "varies"]

    # Given: Stop sequences
    # When: Providing stop tokens
    # Then: Generation stops at token
    def test_chat_with_stop_sequences(self, chat_model: ChatLlamaCpp) -> None:
        messages = [HumanMessage(content="List numbers: 1, 2, 3,")]
        stop = ["5", "done"]
        result = chat_model.invoke(messages, stop=stop)

        content = result.content
        assert "5" not in content
        assert "done" not in content.lower()

    # Given: Streaming enabled
    # When: Using stream()
    # Then: Receives incremental chunks forming valid sentence
    def test_stream_response(self, chat_model: ChatLlamaCpp) -> None:
        messages = [HumanMessage(content="Tell me a 3-word joke.")]
        chunks = list(chat_model.stream(messages))
        assert len(chunks) > 1, "Should yield multiple chunks"
        full_text = "".join(chunk.content for chunk in chunks).strip()
        assert len(full_text) > 0, "Stream should produce non-empty text"
        # Relaxed: just ensure it's joke-related, not exact word count
        joke_keywords = ["joke", "funny", "haha", "lol", "pun"]
        assert any(kw in full_text.lower() for kw in joke_keywords), \
            f"Response should feel like a joke. Got: {full_text[:100]}..."

    # Given: Multiple message types
    # When: Conversation history
    # Then: Context is preserved
    def test_chat_with_history(self, chat_model: ChatLlamaCpp) -> None:
        messages = [
            HumanMessage(content="My name is Alice."),
            AIMessage(content="Hi Alice!"),
            HumanMessage(content="What is my name?"),
        ]
        result = chat_model.invoke(messages)

        assert "alice" in result.content.lower()


class TestChatLlamaCppFunctionalAsync:
    # Given: Async context
    # When: Using ainvoke
    # Then: Returns correct response
    @pytest.mark.asyncio
    async def test_ainvoke(self, chat_model: ChatLlamaCpp) -> None:
        messages = [HumanMessage(content="Say 'async works'")]
        result = await chat_model.ainvoke(messages)

        assert isinstance(result, AIMessage)
        assert "async" in result.content.lower() or "works" in result.content.lower()

    # Given: Async streaming
    # When: Using astream
    # Then: Yields chunks asynchronously
    @pytest.mark.asyncio
    async def test_astream(self, chat_model: ChatLlamaCpp) -> None:
        messages = [HumanMessage(content="Count to 3 slowly.")]
        chunks: List[str] = []
        
        async for chunk in chat_model.astream(messages):
            chunks.append(chunk.content)

        full = "".join(chunks)
        assert any(char.isdigit() for char in full)
        assert len(chunks) >= 3