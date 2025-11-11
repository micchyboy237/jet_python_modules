import pytest
from unittest.mock import MagicMock
from typing import Sequence
from autogen_ext.models.ollama._model_info import _MODEL_INFO
from autogen_core import FunctionCall
from autogen_core.models import (
    UserMessage,
    SystemMessage,
    LLMMessage,
    CreateResult,
    RequestUsage,
    ModelInfo,
)
from autogen_core.tools import ToolSchema
from jet.adapters.autogen.mlx_client import MLXChatCompletionClient, MODEL_MAPPING
from pytest import MonkeyPatch


@pytest.fixture
def mock_mlx_model(monkeypatch: MonkeyPatch):
    # Mock the MLX model and tokenizer
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4])

    def mock_load(model_name: str):
        return mock_model, mock_tokenizer

    def mock_generate(model, tokenizer, prompt: str, max_tokens: int, temp: float, seed: int, stream: bool = False):
        if stream:
            return ["Hello, ", "world!"]
        return "Hello, world!"

    monkeypatch.setattr("mlx_client.load", mock_load)
    monkeypatch.setattr("mlx_client.generate", mock_generate)

    return mock_model, mock_tokenizer


@pytest.fixture
def mlx_client(mock_mlx_model):
    # Initialize MLXChatCompletionClient with a valid model
    return MLXChatCompletionClient(model="llama3.2:3b")


class TestMLXChatCompletionClient:
    @pytest.mark.asyncio
    async def test_text_generation(self, mlx_client: MLXChatCompletionClient):
        # Given: A user message asking for a simple text response
        messages: Sequence[LLMMessage] = [
            UserMessage(content="Say hello world", source="user")
        ]
        expected_content = "Hello, world!"
        expected_usage = RequestUsage(
            prompt_tokens=15,  # Approximate token count for prompt
            completion_tokens=4  # Based on mock tokenizer
        )

        # When: The client generates a response
        result = await mlx_client.create(messages)

        # Then: The response should match the expected content and usage
        assert isinstance(result, CreateResult)
        assert result.content == expected_content
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == expected_usage.prompt_tokens
        assert result.usage.completion_tokens == expected_usage.completion_tokens
        assert result.thought is None

    @pytest.mark.asyncio
    async def test_json_output(self, mlx_client: MLXChatCompletionClient):
        # Given: A user message requesting JSON output
        messages: Sequence[LLMMessage] = [
            SystemMessage(content="Respond in JSON format with a greeting"),
            UserMessage(content="Provide a greeting", source="user")
        ]
        expected_content = """{"greeting": "Hello, world!"}"""
        expected_usage = RequestUsage(
            prompt_tokens=20,  # Approximate token count for prompt
            completion_tokens=4  # Based on mock tokenizer
        )

        # When: The client generates a JSON response
        result = await mlx_client.create(messages, json_output=True)

        # Then: The response should be a valid JSON string
        assert isinstance(result, CreateResult)
        assert result.content == expected_content
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == expected_usage.prompt_tokens
        assert result.usage.completion_tokens == expected_usage.completion_tokens
        assert result.thought is None

    @pytest.mark.asyncio
    async def test_tool_calling(self, mlx_client: MLXChatCompletionClient):
        # Given: A user message with a tool schema
        tools: Sequence[ToolSchema] = [
            {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "properties": {
                        "city": {"type": "string", "description": "The city name"}
                    },
                    "required": ["city"]
                }
            }
        ]
        messages: Sequence[LLMMessage] = [
            UserMessage(content="Get the weather for London", source="user")
        ]
        expected_content = [
            FunctionCall(
                id="0",
                arguments='{"city": "London"}',
                name="get_weather"
            )
        ]
        expected_thought = "I will call the get_weather function for London"
        expected_usage = RequestUsage(
            prompt_tokens=25,  # Approximate token count for prompt + tool schema
            completion_tokens=4  # Based on mock tokenizer
        )

        # Mock the generate function to return a tool call
        def mock_generate_with_tool(*args, **kwargs):
            return f'{{"tool_calls": [{{"name": "get_weather", "arguments": {{"city": "London"}}}}], "thought": "{expected_thought}"}}'

        with MonkeyPatch.context() as m:
            m.setattr("mlx_client.generate", mock_generate_with_tool)

            # When: The client processes a tool call
            result = await mlx_client.create(messages, tools=tools, tool_choice="auto", json_output=True)

            # Then: The response should include the correct tool call
            assert isinstance(result, CreateResult)
            assert result.content == expected_content
            assert result.finish_reason == "tool_calls"
            assert result.thought == expected_thought
            assert result.usage.prompt_tokens == expected_usage.prompt_tokens
            assert result.usage.completion_tokens == expected_usage.completion_tokens

    @pytest.mark.asyncio
    async def test_streaming_response(self, mlx_client: MLXChatCompletionClient):
        # Given: A user message for streaming response
        messages: Sequence[LLMMessage] = [
            UserMessage(content="Stream a greeting", source="user")
        ]
        expected_chunks = ["Hello, ", "world!"]
        expected_content = "Hello, world!"
        expected_usage = RequestUsage(
            prompt_tokens=15,  # Approximate token count for prompt
            completion_tokens=4  # Based on mock tokenizer
        )

        # When: The client streams the response
        chunks = []
        async for chunk in mlx_client.create_stream(messages):
            chunks.append(chunk)

        # Then: The response should stream chunks and end with a CreateResult
        assert len(chunks) == 3  # 2 content chunks + 1 CreateResult
        assert chunks[0] == expected_chunks[0]
        assert chunks[1] == expected_chunks[1]
        assert isinstance(chunks[2], CreateResult)
        assert chunks[2].content == expected_content
        assert chunks[2].finish_reason == "stop"
        assert chunks[2].usage.prompt_tokens == expected_usage.prompt_tokens
        assert chunks[2].usage.completion_tokens == expected_usage.completion_tokens
        assert chunks[2].thought is None

    @pytest.mark.asyncio
    async def test_model_mapping(self, mock_mlx_model):
        # Given: A model name from the mapping
        model_name = "llama3.2:3b"
        expected_mlx_model = MODEL_MAPPING[model_name]
        expected_model_info = ModelInfo(
            name=expected_mlx_model,
            **_MODEL_INFO["llama3.2"]
        )

        # When: The client is initialized with a mapped model
        client = MLXChatCompletionClient(model=model_name)

        # Then: The model name and capabilities should be mapped correctly
        assert client._model_name == expected_mlx_model
        assert client.model_info.name == expected_mlx_model
        assert client.model_info.vision == expected_model_info.vision
        assert client.model_info.function_calling == expected_model_info.function_calling
        assert client.model_info.json_output == expected_model_info.json_output
        assert client.model_info.structured_output == expected_model_info.structured_output
        assert client.model_info.family == expected_model_info.family

    @pytest.mark.asyncio
    async def test_model_info_fallback(self, mock_mlx_model):
        # Given: An unmapped model name not in _MODEL_INFO
        model_name = "unknown_model:1b"
        expected_mlx_model = model_name  # No mapping, uses input directly
        expected_model_info = ModelInfo(
            name=expected_mlx_model,
            family="mlx",
            vision=False,
            function_calling=True,
            json_output=True,
            token_limit=32768,
            structured_output=True,
        )

        # When: The client is initialized with an unmapped model
        client = MLXChatCompletionClient(model=model_name)

        # Then: The model should use fallback model info
        assert client._model_name == expected_mlx_model
        assert client.model_info.name == expected_mlx_model
        assert client.model_info.vision == expected_model_info.vision
        assert client.model_info.function_calling == expected_model_info.function_calling
        assert client.model_info.json_output == expected_model_info.json_output
        assert client.model_info.structured_output == expected_model_info.structured_output
        assert client.model_info.family == expected_model_info.family
        assert client.model_info.token_limit == expected_model_info.token_limit

    def teardown_method(self):
        # Clean up any logged files if necessary
        import shutil
        log_dir = Path(DEFAULT_OLLAMA_LOG_DIR)
        if log_dir.exists():
            shutil.rmtree(log_dir)
