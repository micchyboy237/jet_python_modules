#!/usr/bin/env python3
"""Example demonstrating initializing McpWorkbench with a ChatCompletionClient."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage

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

async def initialize_workbench_with_model() -> Optional[McpWorkbench]:
    """
    Initialize McpWorkbench with a custom ChatCompletionClient.

    Returns:
        Optional[McpWorkbench]: Initialized workbench or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    model_client = SimpleChatCompletionClient()
    
    try:
        workbench = McpWorkbench(server_params=server_params, model_client=model_client)
        await workbench.start()
        print(f"Workbench initialized with model: {workbench._model_client.model_info['family']}")
        return workbench
    except Exception as e:
        print(f"Error initializing workbench: {e}")
        return None
    finally:
        if 'workbench' in locals():
            await workbench.stop()

async def main() -> None:
    """Main function to run the initialize workbench example."""
    workbench = await initialize_workbench_with_model()
    if not workbench:
        print("Failed to initialize workbench.")

if __name__ == "__main__":
    asyncio.run(main())