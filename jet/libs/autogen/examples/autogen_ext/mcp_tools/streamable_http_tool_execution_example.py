#!/usr/bin/env python3
"""Example demonstrating executing a tool using StreamableHttpMcpToolAdapter."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import StreamableHttpMcpToolAdapter, StreamableHttpServerParams
from mcp import Tool
from mcp.types import TextContent

async def execute_streamable_http_tool() -> List[Any]:
    """
    Execute a test StreamableHttp tool using StreamableHttpMcpToolAdapter.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = StreamableHttpServerParams(url="http://localhost:8080")
    tool = Tool(
        name="test_streamable_http_tool",
        description="A test StreamableHttp tool for demonstration",
        inputSchema={
            "type": "object",
            "properties": {"test_param": {"type": "string"}},
            "required": ["test_param"],
        },
    )
    adapter = StreamableHttpMcpToolAdapter(server_params=server_params, tool=tool)
    
    try:
        result = await adapter.run_json(
            args={"test_param": "http_input"},
            cancellation_token=CancellationToken(),
        )
        print("StreamableHttp Tool Execution Result:")
        for content in result:
            if isinstance(content, TextContent):
                print(f"Text: {content.text}")
        return result
    except Exception as e:
        print(f"Error executing StreamableHttp tool: {e}")
        return []
    finally:
        await adapter.close()

async def main() -> None:
    """Main function to run the StreamableHttp tool execution example."""
    result = await execute_streamable_http_tool()
    if not result:
        print("No result returned from StreamableHttp tool execution.")

if __name__ == "__main__":
    asyncio.run(main())