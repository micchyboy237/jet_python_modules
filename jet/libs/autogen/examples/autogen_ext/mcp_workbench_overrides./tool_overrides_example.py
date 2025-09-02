#!/usr/bin/env python3
"""Example demonstrating listing tools with overrides using McpWorkbench."""
import asyncio
from typing import List
from autogen_core.tools import ToolOverride
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def list_tools_with_overrides() -> List[dict]:
    """
    List available tools with name and description overrides using McpWorkbench.

    Returns:
        List[dict]: List of tool dictionaries with overridden names and descriptions.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    overrides = {
        "fetch": ToolOverride(name="web_fetch", description="Enhanced web fetching tool"),
        "search": ToolOverride(description="Advanced search functionality"),
    }
    workbench = McpWorkbench(server_params=server_params, tool_overrides=overrides)
    
    try:
        tools = await workbench.list_tools()
        print("Available Tools with Overrides:")
        for tool in tools:
            print(f"- {tool['name']}: {tool['description']}")
        return tools
    except Exception as e:
        print(f"Error listing tools: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the tool overrides example."""
    tools = await list_tools_with_overrides()
    if not tools:
        print("No tools returned.")

if __name__ == "__main__":
    asyncio.run(main())