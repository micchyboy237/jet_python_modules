#!/usr/bin/env python3
"""Example demonstrating calling a tool with a model client using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import CallToolResult
from autogen_core.models import ChatCompletionClient, CreateResult, ModelInfo, RequestUsage
from jet.libs.autogen.ollama_client import OllamaChatCompletionClient
from jet.servers.mcp.config import MCP_SERVER_PATH


async def call_test_tool(param: str) -> Optional[CallToolResult]:
    """
    Call a test tool with a parameter using McpWorkbench and a model client.

    Args:
        param (str): Parameter for the tool.

    Returns:
        Optional[CallToolResult]: The tool execution result or None if failed.
    """
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/temp/docs.llamaindex.ai/en/latest/workflows/v2/index.html"

    server_params = StdioServerParams(
        command="python", args=[MCP_SERVER_PATH])
    model_client = OllamaChatCompletionClient(model="llama3.2")
    workbench = McpWorkbench(
        server_params=server_params, model_client=model_client)

    try:
        await workbench.start()
        result: CallToolResult = await workbench.call_tool("read_file", {"arguments": {"file_path": file_path, "encoding": "utf-8"}})
        print(f"Tool Result ({result.name}):")
        for content in result.result:
            print(f"Content: {content.content}")
        return result
    except Exception as e:
        print(f"Error calling tool: {e}")
        return None
    finally:
        await workbench.stop()


async def main() -> None:
    """Main function to run the call tool example."""
    result = await call_test_tool(param="example_value")
    if not result:
        print("Failed to call tool.")

if __name__ == "__main__":
    asyncio.run(main())
