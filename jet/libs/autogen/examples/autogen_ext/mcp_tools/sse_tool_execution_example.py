#!/usr/bin/env python3
"""Example demonstrating executing a tool using SseMcpToolAdapter."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import SseMcpToolAdapter, SseServerParams
from mcp import Tool
from mcp.types import TextContent

async def execute_sse_tool() -> List[Any]:
    """
    Execute a test SSE tool using SseMcpToolAdapter.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = SseServerParams(url="http://localhost:8080")
    tool = Tool(
        name="test_sse_tool",
        description="A test SSE tool for demonstration",
        inputSchema={
            "type": "object",
            "properties": {"test_param": {"type": "string"}},
            "required": ["test_param"],
        },
    )
    adapter = SseMcpToolAdapter(server_params=server_params, tool=tool)
    
    try:
        result = await adapter.run_json(
            args={"test_param": "sse_input"},
            cancellation_token=CancellationToken(),
        )
        print("SSE Tool Execution Result:")
        for content in result:
            if isinstance(content, TextContent):
                print(f"Text: {content.text}")
        return result
    except Exception as e:
        print(f"Error executing SSE tool: {e}")
        return []
    finally:
        await adapter.close()

async def main() -> None:
    """Main function to run the SSE tool execution example."""
    result = await execute_sse_tool()
    if not result:
        print("No result returned from SSE tool execution.")

if __name__ == "__main__":
    asyncio.run(main())