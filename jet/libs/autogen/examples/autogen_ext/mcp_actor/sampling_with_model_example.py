#!/usr/bin/env python3
"""Example demonstrating sampling with a model client using McpSessionActor."""
import asyncio
from pathlib import Path
from typing import Optional
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from autogen_ext.tools.mcp import McpSessionActor, StdioServerParams
from mcp.types import CreateMessageRequestParams, CreateMessageResult, SamplingMessage, TextContent

class SimpleChatCompletionClient(ChatCompletionClient):
    """Simple chat completion client for demonstration."""
    def __init__(self):
        self._model_info = ModelInfo(
            vision=False,
            function_calling=False,
            json_output=False,
            family="simple-model",
            structured_output=False,
        )

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    async def create(self, messages, **kwargs) -> CreateResult:
        return CreateResult(
            content="Hello, I'm a simple model response!",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
            cached=False,
        )

    async def close(self) -> None:
        pass

async def perform_sampling() -> Optional[CreateMessageResult]:
    """
    Perform sampling with a model client using McpSessionActor.

    Returns:
        Optional[CreateMessageResult]: Sampling result or None if failed.
    """
    server_path = Path(__file__).parent / "mcp_server_comprehensive.py"
    server_params = StdioServerParams(
        command="uv",
        args=["run", "python", str(server_path)],
        read_timeout_seconds=10,
    )
    model_client = SimpleChatCompletionClient()
    actor = McpSessionActor(server_params=server_params, model_client=model_client)
    
    try:
        await actor.initialize()
        params = CreateMessageRequestParams(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text="Hello from test"))],
            maxTokens=100,
            systemPrompt="You are a helpful assistant.",
        )
        result = await actor._sampling_callback(context=None, params=params)
        if isinstance(result, CreateMessageResult):
            print(f"Sampling Result: {result.content.text} (Model: {result.model})")
            return result
        else:
            print(f"Sampling Error: {result.message}")
            return None
    except Exception as e:
        print(f"Error performing sampling: {e}")
        return None
    finally:
        await actor.close()

async def main() -> None:
    """Main function to run the sampling with model example."""
    result = await perform_sampling()
    if not result:
        print("No result returned from sampling.")

if __name__ == "__main__":
    asyncio.run(main())