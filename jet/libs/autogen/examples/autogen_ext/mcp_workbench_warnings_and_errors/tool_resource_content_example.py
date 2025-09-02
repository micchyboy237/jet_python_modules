#!/usr/bin/env python3
"""Example demonstrating handling embedded resource content from tool execution with McpWorkbench."""
import asyncio
from typing import Optional
from autogen_core import CancellationToken
from autogen_core.tools import TextResultContent
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import CallToolResult

async def execute_tool_with_resource_content() -> Optional[CallToolResult]:
    """
    Execute a tool that returns embedded resource content using McpWorkbench.

    Returns:
        Optional[CallToolResult]: Tool execution result or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        await workbench.start()
        result = await workbench.call_tool("resource_tool", cancellation_token=CancellationToken())
        if not result.is_error:
            for content in result.result:
                if isinstance(content, TextResultContent):
                    print(f"Resource Content: {content.content}")
        return result
    except Exception as e:
        print(f"Error executing tool: {e}")
        return None
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the tool resource content example."""
    result = await execute_tool_with_resource_content()
    if not result:
        print("No result returned from tool execution.")

if __name__ == "__main__":
    asyncio.run(main())