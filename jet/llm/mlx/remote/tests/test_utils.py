import pytest
from typing import List, Dict, Optional, Union, Literal, Iterator, Callable
from unittest.mock import Mock, patch
from jet.llm.mlx.remote.types import (
    Message,
    ChatCompletionRequest,
    ChatCompletionResponse,
    TextCompletionRequest,
    TextCompletionResponse,
    ToolCall,
)
from jet.llm.mlx.chat_history import ChatHistory
from jet.llm.mlx.remote.utils import (
    prepare_chat_request,
    prepare_text_request,
    process_chat_response,
    process_stream_chat_response,
    process_text_response,
    process_stream_text_response,
)


@pytest.fixture
def chat_history():
    """Fixture to provide a fresh ChatHistory instance."""
    return ChatHistory()


@pytest.fixture
def mock_tool():
    """Fixture to provide a mock callable tool."""
    tool = Mock(spec=Callable)
    tool.__name__ = "test_tool"
    return tool


@pytest.fixture
def mock_get_method_info():
    """Fixture to mock get_method_info function."""
    with patch("jet.llm.mlx.remote.utils.get_method_info") as mock:
        mock.return_value = {"name": "test_tool", "parameters": {}}
        yield mock


@pytest.fixture
def mock_has_tools():
    """Fixture to mock has_tools function."""
    with patch("jet.llm.mlx.remote.utils.has_tools") as mock:
        mock.return_value = True
        yield mock


@pytest.fixture
def mock_parse_and_evaluate():
    """Fixture to mock parse_and_evaluate function."""
    with patch("jet.llm.mlx.remote.utils.parse_and_evaluate") as mock:
        yield mock


@pytest.fixture
def mock_execute_tool_calls():
    """Fixture to mock execute_tool_calls function."""
    with patch("jet.llm.mlx.remote.utils.execute_tool_calls") as mock:
        mock.return_value = [
            {"tool_call": {"function": {"name": "test_tool"}}, "tool_result": "result"}]
        yield mock


@pytest.fixture
def mock_process_response_format():
    """Fixture to mock process_response_format function."""
    with patch("jet.llm.mlx.remote.utils.process_response_format") as mock:
        mock.side_effect = lambda x, fmt: x  # Return input as-is for simplicity
        yield mock


class TestUtils:
    """Tests for utility functions in jet.llm.mlx.remote.utils."""

    def test_prepare_chat_request_with_string_message(
        self, chat_history, mock_tool, mock_get_method_info, mock_has_tools, mock_process_response_format
    ):
        """Test prepare_chat_request with a string message."""
        # Given: A string message and basic parameters
        message = "Hello, world!"
        system_prompt = "You are a helpful assistant."
        model = "test_model"
        expected: ChatCompletionRequest = {
            "messages": [{"role": "system", "content": system_prompt, "tool_calls": None},
                         {"role": "user", "content": message, "tool_calls": None}],
            "model": model,
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
        }

        # When: Preparing the chat request
        result = prepare_chat_request(
            messages=message,
            history=chat_history,
            system_prompt=system_prompt,
            with_history=False,
            response_format="text",
            model=model,
            tools=[mock_tool]
        )

        # Then: The request matches the expected structure
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_prepare_chat_request_with_message_list(
        self, chat_history, mock_get_method_info, mock_has_tools, mock_process_response_format
    ):
        """Test prepare_chat_request with a list of messages."""
        # Given: A list of messages with tool calls
        messages: List[Message] = [
            {"role": "user", "content": "Run test_tool", "tool_calls": [
                {"type": "function", "function": {"name": "test_tool", "arguments": {}}}
            ]}
        ]
        model = "test_model"
        expected: ChatCompletionRequest = {
            "messages": messages,
            "model": model,
            "stream": False,
            "tools": [{"type": "function", "function": {"name": "test_tool", "parameters": {}}}]
        }

        # When: Preparing the chat request
        result = prepare_chat_request(
            messages=messages,
            history=chat_history,
            system_prompt=None,
            with_history=False,
            response_format="json",
            model=model,
            tools=[Mock(spec=Callable)]
        )

        # Then: The request matches the expected structure
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_prepare_text_request(
        self, mock_process_response_format
    ):
        """Test prepare_text_request with a prompt."""
        # Given: A prompt and basic parameters
        prompt = "Generate a story"
        model = "test_model"
        expected: TextCompletionRequest = {
            "prompt": prompt,
            "model": model,
            "stream": False
        }

        # When: Preparing the text request
        result = prepare_text_request(
            prompt=prompt,
            response_format="text",
            model=model
        )

        # Then: The request matches the expected structure
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_chat_response_with_content(
        self, chat_history, mock_parse_and_evaluate, mock_execute_tool_calls
    ):
        """Test process_chat_response with content and no tool calls."""
        # Given: A chat response with content
        response: ChatCompletionResponse = {
            "id": "123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test_model",
            "system_fingerprint": "fp_123",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!", "tool_calls": None},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        expected: ChatCompletionResponse = response.copy()
        expected["content"] = "Hello!"
        expected["tool_calls"] = None
        expected["history"] = [{"role": "assistant",
                                "content": "Hello!", "tool_calls": None}]

        # When: Processing the chat response with history
        result = process_chat_response(
            response, chat_history, with_history=True, tools=None)

        # Then: The response is processed correctly with history
        assert result == expected, f"Expected {expected}, but got {result}"
        assert chat_history.get_messages(
        ) == expected["history"], "History not updated correctly"

    def test_process_chat_response_with_tool_calls(
        self, chat_history, mock_parse_and_evaluate, mock_execute_tool_calls
    ):
        """Test process_chat_response with tool calls."""
        # Given: A chat response with tool calls
        response: ChatCompletionResponse = {
            "id": "123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test_model",
            "system_fingerprint": "fp_123",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"type": "function", "function": {"name": "test_tool", "arguments": {}}}]
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        expected: ChatCompletionResponse = response.copy()
        expected["content"] = ""
        expected["tool_calls"] = [{"type": "function",
                                   "function": {"name": "test_tool", "arguments": {}}}]
        expected["tool_execution"] = mock_execute_tool_calls.return_value
        expected["history"] = [
            {"role": "assistant", "content": "", "tool_calls": expected["tool_calls"]}
        ]

        # When: Processing the chat response with tools and history
        result = process_chat_response(
            response, chat_history, with_history=True, tools=[Mock(spec=Callable)])

        # Then: The response is processed correctly with tool calls and history
        assert result == expected, f"Expected {expected}, but got {result}"
        assert chat_history.get_messages(
        ) == expected["history"], "History not updated correctly"

    def test_process_stream_chat_response(
        self, chat_history, mock_parse_and_evaluate, mock_execute_tool_calls
    ):
        """Test process_stream_chat_response with streaming chunks."""
        # Given: A stream of chat response chunks
        chunks = [
            {
                "id": "123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hello"},
                        "finish_reason": None
                    }
                ]
            },
            {
                "id": "123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " world!"},
                        "finish_reason": "stop"
                    }
                ]
            }
        ]
        expected_chunks = [
            {
                "id": "123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello", "tool_calls": None},
                        "finish_reason": None
                    }
                ]
            },
            {
                "id": "123",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": " world!", "tool_calls": None},
                        "finish_reason": "stop"
                    }
                ],
                "content": "Hello world!",
                "tool_calls": None,
                "history": [{"role": "assistant", "content": "Hello world!", "tool_calls": None}]
            }
        ]

        # When: Processing the stream of chat response chunks
        result_chunks = list(process_stream_chat_response(
            chunks, chat_history, with_history=True, tools=None))

        # Then: The chunks are processed correctly with aggregated content and history
        assert result_chunks == expected_chunks, f"Expected {expected_chunks}, but got {result_chunks}"
        assert chat_history.get_messages(
        ) == expected_chunks[1]["history"], "History not updated correctly"

    def test_process_text_response(self):
        """Test process_text_response with content."""
        # Given: A text completion response
        response: TextCompletionResponse = {
            "id": "123",
            "object": "text_completion",
            "created": 1234567890,
            "model": "test_model",
            "system_fingerprint": "fp_123",
            "choices": [
                {"index": 0, "text": "Once upon a time", "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        expected: TextCompletionResponse = response.copy()
        expected["content"] = "Once upon a time"

        # When: Processing the text response
        result = process_text_response(response)

        # Then: The response is processed correctly with aggregated content
        assert result == expected, f"Expected {expected}, but got {result}"

    def test_process_stream_text_response(self):
        """Test process_stream_text_response with streaming chunks."""
        # Given: A stream of text completion chunks
        chunks = [
            {
                "id": "123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {"index": 0, "text": "Once upon", "finish_reason": None}
                ]
            },
            {
                "id": "123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {"index": 0, "text": " a time", "finish_reason": "stop"}
                ]
            }
        ]
        expected_chunks = [
            {
                "id": "123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {"index": 0, "text": "Once upon", "finish_reason": None}
                ]
            },
            {
                "id": "123",
                "object": "text_completion",
                "created": 1234567890,
                "model": "test_model",
                "system_fingerprint": "fp_123",
                "choices": [
                    {"index": 0, "text": " a time", "finish_reason": "stop"}
                ],
                "content": "Once upon a time"
            }
        ]

        # When: Processing the stream of text response chunks
        result_chunks = list(process_stream_text_response(chunks))

        # Then: The chunks are processed correctly with aggregated content
        assert result_chunks == expected_chunks, f"Expected {expected_chunks}, but got {result_chunks}"
