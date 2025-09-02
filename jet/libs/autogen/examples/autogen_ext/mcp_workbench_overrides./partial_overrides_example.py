#!/usr/bin/env python3
"""Example demonstrating partial tool overrides using McpWorkbench."""
import asyncio
from typing import List
from autogen_core.tools import ToolOverride
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def list_tools_with_partial_overrides() -> List[dict]:
    """
    List tools with partial overrides (name or description only) using McpWorkbench.

    Returns:
        List[dict]: List of tool dictionaries with partial overrides.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    overrides = {
        "fetch": ToolOverride(name="web_fetch"),
        "search": ToolOverride(description="Advanced search"),
    }
    workbench = McpWorkbench(server_params=server_params, tool_overrides=overrides)
    
    try:
        tools = await workbench.list_tools()
        print("Available Tools with Partial Overrides:")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        return tools
    except Exception as e:
        print(f"Error listing tools: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the partial overrides example."""
    tools = await list_tools_with_partial_overrides()
    if not tools:
        print("No tools returned.")

if __name__ == "__main__":
    asyncio.run(main())