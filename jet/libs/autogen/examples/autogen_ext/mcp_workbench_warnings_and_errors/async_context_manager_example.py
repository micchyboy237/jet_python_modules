#!/usr/bin/env python3
"""Example demonstrating async context manager usage with McpWorkbench."""
import asyncio
from typing import List
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def list_tools_with_context_manager() -> List[dict]:
    """
    List tools using McpWorkbench within an async context manager.

    Returns:
        List[dict]: List of available tools.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    async with workbench:
        try:
            tools = await workbench.list_tools()
            print("Available Tools:", [tool["name"] for tool in tools])
            return tools
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []

async def main() -> None:
    """Main function to run the async context manager example."""
    tools = await list_tools_with_context_manager()
    if not tools:
        print("No tools returned from context manager.")

if __name__ == "__main__":
    asyncio.run(main())