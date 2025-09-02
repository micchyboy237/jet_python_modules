#!/usr/bin/env python3
"""Example demonstrating executing a tool using StdioMcpToolAdapter."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import StdioMcpToolAdapter, StdioServerParams
from mcp import Tool
from mcp.types import TextContent

async def execute_stdio_tool() -> List[Any]:
    """
    Execute a test tool using StdioMcpToolAdapter.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    tool = Tool(
        name="test_tool",
        description="A test tool for demonstration",
        inputSchema={
            "type": "object",
            "properties": {"test_param": {"type": "string"}},
            "required": ["test_param"],
        },
    )
    adapter = StdioMcpToolAdapter(server_params=server_params, tool=tool)
    
    try:
        result = await adapter.run_json(
            args={"test_param": "example_input"},
            cancellation_token=CancellationToken(),
        )
        print("Tool Execution Result:")
        for content in result:
            if isinstance(content, TextContent):
                print(f"Text: {content.text}")
        return result
    except Exception as e:
        print(f"Error executing tool: {e}")
        return []
    finally:
        await adapter.close()

async def main() -> None:
    """Main function to run the stdio tool execution example."""
    result = await execute_stdio_tool()
    if not result:
        print("No result returned from tool execution.")

if __name__ == "__main__":
    asyncio.run(main())