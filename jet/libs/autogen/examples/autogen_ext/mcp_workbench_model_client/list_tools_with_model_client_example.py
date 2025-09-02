#!/usr/bin/env python3
"""Example demonstrating listing tools with a model client using McpWorkbench."""
import asyncio
from typing import List, Dict
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
            content="Tool list response",
            finish_reason="stop",
            usage=RequestUsage(prompt_tokens=10, completion_tokens=5),
            cached=False,
        )

    async def close(self) -> None:
        pass

async def list_tools_with_model() -> List[Dict]:
    """
    List available tools using McpWorkbench with a model client.

    Returns:
        List[Dict]: List of available tools.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    model_client = SimpleChatCompletionClient()
    workbench = McpWorkbench(server_params=server_params, model_client=model_client)
    
    try:
        await workbench.start()
        tools = await workbench.list_tools()
        print("Available Tools:")
        for tool in tools:
            print(f"- {tool.get('name')}: {tool.get('description')}")
        return tools
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the list tools example."""
    tools = await list_tools_with_model()
    if not tools:
        print("No tools found.")

if __name__ == "__main__":
    asyncio.run(main())