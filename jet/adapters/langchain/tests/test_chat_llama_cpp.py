# test_chat_llama_cpp.py
"""Unit tests for ChatLlamaCpp class."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from pydantic import BaseModel, Field

from jet.adapters.langchain.chat_llama_cpp import ChatLlamaCpp

@pytest.fixture
def llm():
    """Fixture for ChatLlamaCpp instance with cleanup."""
    llm = ChatLlamaCpp(
        model="gpt-oss:20b",
        base_url="http://shawn-pc.local:8080/v1",
        temperature=0.8,
        max_tokens=128,
    )
    yield llm
    # Cleanup: No explicit cleanup needed as client connections are managed by OpenAI client

@pytest.fixture
def mock_client():
    """Fixture for mocking OpenAI client."""
    client = Mock()
    client.chat.completions.create = Mock()
    yield client

@pytest.fixture
def mock_async_client():
    """Fixture for mocking OpenAI async client."""
    async_client = AsyncMock()
    async_client.chat.completions.create = AsyncMock()
    yield async_client

class TestChatLlamaCppInitialization:
    """Tests for ChatLlamaCpp initialization."""

    def test_initialization_with_valid_params(self):
        """Test: Initialize ChatLlamaCpp with valid parameters."""
        # Given: Valid initialization parameters
        params = {
            "model": "gpt-oss:20b",
            "temperature": 0.8,
            "max_tokens": 128,
            "base_url": "http://shawn-pc.local:8080/v1",
        }

        # When: Initializing the LLM
        llm = ChatLlamaCpp(**params)

        # Then: Verify attributes are set correctly
        result = {
            "model": llm.model,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "base_url": llm.base_url,
        }
        expected = params
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_initialization_with_validation_failure(self, mock_client):
        """Test: Handle server validation failure during initialization."""
        # Given: A mock client that raises an error on model list
        mock_client.models.list.side_effect = Exception("Server unreachable")
        with patch("chat_llama_cpp.Client", return_value=mock_client):
            # When: Initializing with validate_model_on_init=True
            with pytest.raises(ValueError) as exc_info:
                ChatLlamaCpp(model="gpt-oss:20b", validate_model_on_init=True)

            # Then: Verify the error message
            result = str(exc_info.value)
            expected = "Failed to validate model on server: Server unreachable"
            assert result == expected, f"Expected error message {expected}, but got {result}"

class TestChatLlamaCppMessageConversion:
    """Tests for message conversion to OpenAI format."""

    def test_convert_messages_to_openai_format(self, llm):
        """Test: Convert LangChain messages to OpenAI-compatible format."""
        # Given: A list of mixed message types
        messages = [
            SystemMessage(content="You are a translator."),
            HumanMessage(content="Translate 'Hello' to French."),
            AIMessage(content="Bonjour", tool_calls=[
                ToolCall(name="translate", args={"text": "Hello", "language": "French"}, id="call_123")
            ]),
            ToolMessage(content="Translation completed", tool_call_id="call_123"),
        ]

        # When: Converting messages
        result = llm._convert_messages_to_openai_messages(messages)

        # Then: Verify the converted messages
        expected = [
            {"role": "system", "content": "You are a translator."},
            {"role": "user", "content": "Translate 'Hello' to French."},
            {
                "role": "assistant",
                "content": "Bonjour",
                "tool_calls": [
                    {
                        "type": "function",
                        "id": "call_123",
                        "function": {"name": "translate", "arguments": {"text": "Hello", "language": "French"}},
                    }
                ],
            },
            {"role": "tool", "content": "Translation completed", "tool_call_id": "call_123"},
        ]
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_convert_messages_with_image(self, llm):
        """Test: Convert messages with image content."""
        # Given: A message with image content
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
            ])
        ]

        # When: Converting messages
        result = llm._convert_messages_to_openai_messages(messages)

        # Then: Verify the converted message
        expected = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abc123"}},
                ],
            }
        ]
        assert result == expected, f"Expected {expected}, but got {result}"

class TestChatLlamaCppInvocation:
    """Tests for synchronous invocation."""

    def test_invoke(self, llm, mock_client):
        """Test: Invoke the model with a simple message."""
        # Given: Mock client response
        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(content="Bonjour", role="assistant")
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_client", mock_client):
            messages = [HumanMessage(content="Translate 'Hello' to French.")]

            # When: Invoking the model
            result = llm._generate(messages).generations[0].message

            # Then: Verify the response
            expected = AIMessage(
                content="Bonjour",
                additional_kwargs={},
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
            assert result.content == expected.content, f"Expected content {expected.content}, but got {result.content}"
            assert result.usage_metadata == expected.usage_metadata, f"Expected usage {expected.usage_metadata}, but got {result.usage_metadata}"

class TestChatLlamaCppStreaming:
    """Tests for synchronous streaming."""

    def test_stream(self, llm, mock_client):
        """Test: Stream response from the model."""
        # Given: Mock streamed chunks
        chunks = [
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="Hel"), finish_reason=None, index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="lo"), finish_reason=None, index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="!"), finish_reason="stop", index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            ),
        ]
        mock_client.chat.completions.create.return_value = iter(chunks)
        with patch.object(llm, "_client", mock_client):
            messages = [HumanMessage(content="Say hello!")]

            # When: Streaming the response
            result = [chunk.text for chunk in llm._stream(messages)]

            # Then: Verify the streamed content
            expected = ["Hel", "lo", "!"]
            assert result == expected, f"Expected {expected}, but got {result}"

class TestChatLlamaCppAsync:
    """Tests for asynchronous operations."""

    @pytest.mark.asyncio
    async def test_ainvoke(self, llm, mock_async_client):
        """Test: Async invoke the model."""
        # Given: Mock async client response
        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(content="Bonjour", role="assistant")
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_async_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_async_client", mock_async_client):
            messages = [HumanMessage(content="Translate 'Hello' to French.")]

            # When: Invoking async
            result = (await llm._agenerate(messages)).generations[0].message

            # Then: Verify the response
            expected = AIMessage(
                content="Bonjour",
                additional_kwargs={},
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
            assert result.content == expected.content, f"Expected content {expected.content}, but got {result.content}"
            assert result.usage_metadata == expected.usage_metadata, f"Expected usage {expected.usage_metadata}, but got {result.usage_metadata}"

    @pytest.mark.asyncio
    async def test_astream(self, llm, mock_async_client):
        """Test: Async stream response from the model."""
        # Given: Mock async streamed chunks
        chunks = [
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="Hel"), finish_reason=None, index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="lo"), finish_reason=None, index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
            ),
            ChatCompletionChunk(
                id="run-123",
                choices=[Choice(delta=ChoiceDelta(content="!"), finish_reason="stop", index=0)],
                created=1698162061,
                model="gpt-oss:20b",
                object="chat.completion.chunk",
                usage=CompletionUsage(prompt_tokens=10, completion_tokens=3, total_tokens=13),
            ),
        ]
        mock_async_client.chat.completions.create.return_value = chunks
        with patch.object(llm, "_async_client", mock_async_client):
            messages = [HumanMessage(content="Say hello!")]

            # When: Async streaming
            result = [chunk.text async for chunk in llm._astream(messages)]

            # Then: Verify the streamed content
            expected = ["Hel", "lo", "!"]
            assert result == expected, f"Expected {expected}, but got {result}"

class TestChatLlamaCppJSONMode:
    """Tests for JSON mode output."""

    def test_json_mode(self, llm, mock_client):
        """Test: Invoke with JSON mode."""
        # Given: Mock client response in JSON format
        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(content='{"location": "Pune", "time_of_day": "morning"}', role="assistant")
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        )
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_client", mock_client):
            llm.format = "json"
            messages = [
                HumanMessage(content="Return a query for weather with keys: location, time_of_day in JSON.")
            ]

            # When: Invoking with JSON mode
            result = llm._generate(messages).generations[0].message.content

            # Then: Verify the JSON response
            expected = '{"location": "Pune", "time_of_day": "morning"}'
            assert result == expected, f"Expected {expected}, but got {result}"

class TestChatLlamaCppToolCalling:
    """Tests for tool calling functionality."""

    def test_tool_calling(self, llm, mock_client):
        """Test: Invoke with tool calling."""
        # Given: A Pydantic tool and mock response
        class Multiply(BaseModel):
            a: int = Field(..., description="First integer")
            b: int = Field(..., description="Second integer")

        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(
                    content="",
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "Multiply",
                                "arguments": json.dumps({"a": 45, "b": 67}),
                            },
                        }
                    ],
                )
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_client", mock_client):
            messages = [HumanMessage(content="What is 45*67")]

            # When: Invoking with bound tools
            llm_with_tools = llm.bind_tools([Multiply], tool_choice="Multiply")
            result = llm_with_tools._generate(messages).generations[0].message.tool_calls

            # Then: Verify the tool call
            expected = [
                {
                    "id": "call_123",
                    "name": "Multiply",
                    "args": {"a": 45, "b": 67},
                    "type": "tool_call",
                }
            ]
            assert result == expected, f"Expected {expected}, but got {result}"

class TestChatLlamaCppReasoning:
    """Tests for reasoning mode."""

    def test_reasoning_mode(self, llm, mock_client):
        """Test: Invoke with reasoning enabled."""
        # Given: Mock response with reasoning
        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(content="The word 'strawberry' contains **three r letters**.", role="assistant")
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_client", mock_client):
            messages = [HumanMessage(content="How many r's in the word strawberry?")]
            llm.reasoning = True

            # When: Invoking with reasoning
            result = llm._generate(messages).generations[0].message

            # Then: Verify reasoning content
            expected_content = "The word 'strawberry' contains **three r letters**."
            assert result.content == expected_content, f"Expected content {expected_content}, but got {result.content}"
            # Note: Reasoning content is mocked in additional_kwargs during streaming, tested in stream test

class TestChatLlamaCppStructuredOutput:
    """Tests for structured output functionality."""

    def test_structured_output(self, llm, mock_client):
        """Test: Invoke with structured output."""
        # Given: A Pydantic schema and mock response
        class AnswerWithJustification(BaseModel):
            answer: str
            justification: str

        mock_response = ChatCompletion(
            id="run-123",
            choices=[
                ChatCompletionMessage(
                    content="",
                    role="assistant",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "AnswerWithJustification",
                                "arguments": json.dumps({
                                    "answer": "They weigh the same",
                                    "justification": "Both a pound of bricks and a pound of feathers weigh one pound."
                                }),
                            },
                        }
                    ],
                )
            ],
            created=1698162061,
            model="gpt-oss:20b",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )
        mock_client.chat.completions.create.return_value = mock_response
        with patch.object(llm, "_client", mock_client):
            messages = [HumanMessage(content="What weighs more: a pound of bricks or a pound of feathers?")]
            structured_llm = llm.with_structured_output(AnswerWithJustification, method="function_calling")

            # When: Invoking with structured output
            result = structured_llm.invoke(messages)

            # Then: Verify the structured output
            expected = AnswerWithJustification(
                answer="They weigh the same",
                justification="Both a pound of bricks and a pound of feathers weigh one pound."
            )
            assert result == expected, f"Expected {expected}, but got {result}"