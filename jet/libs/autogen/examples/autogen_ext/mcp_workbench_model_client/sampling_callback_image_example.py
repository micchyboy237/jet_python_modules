#!/usr/bin/env python3
"""Example demonstrating invoking sampling callback with image content using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from mcp.types import CreateMessageRequestParams, CreateMessageResult, SamplingMessage, ImageContent

class VisionChatCompletionClient(ChatCompletionClient):
    """Vision-capable chat completion client for demonstration."""
    def __init__(self):
        self._model_info = ModelInfo(
            vision=True,
            function_calling=False,
            json_output=False,
            family="vision-model",
            structured_output=False,
        )

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    async def create(self, messages, **kwargs) -> CreateResult:
        return CreateResult(
            content="Image analyzed: A small pixel image.",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
            cached=False,
        )

    async def close(self) -> None:
        pass

async def invoke_sampling_callback_with_image() -> Optional[CreateMessageResult]:
    """
    Invoke sampling callback with an image message.

    Returns:
        Optional[CreateMessageResult]: The sampling result or None if failed.
    """
    from autogen_ext.tools.mcp._actor import McpSessionActor
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    model_client = VisionChatCompletionClient()
    actor = McpSessionActor(server_params=server_params, model_client=model_client)
    
    try:
        await actor.initialize()
        test_image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        params = CreateMessageRequestParams(
            messages=[
                SamplingMessage(
                    role="user",
                    content=ImageContent(type="image", data=test_image_b64, mimeType="image/png")
                )
            ],
            maxTokens=100,
        )
        result = await actor._sampling_callback(context=None, params=params)
        if isinstance(result, CreateMessageResult):
            print(f"Sampling Result: {result.content.text} (Model: {result.model})")
            return result
        else:
            print(f"Sampling Error: {result.message}")
            return None
    finally:
        await actor.close()

async def main() -> None:
    """Main function to run the sampling callback with image example."""
    result = await invoke_sampling_callback_with_image()
    if not result:
        print("Failed to invoke sampling callback.")

if __name__ == "__main__":
    asyncio.run(main())