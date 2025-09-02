#!/usr/bin/env python3
"""Example demonstrating calling a tool with an overridden name using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_core.tools import ToolOverride
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import CallToolResult

async def call_tool_with_override() -> Optional[CallToolResult]:
    """
    Call the 'web_fetch' tool (overridden from 'fetch') using McpWorkbench.

    Returns:
        Optional[CallToolResult]: Tool execution result or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    overrides = {
        "fetch": ToolOverride(name="web_fetch", description="Enhanced web fetching tool"),
    }
    workbench = McpWorkbench(server_params=server_params, tool_overrides=overrides)
    
    try:
        result: CallToolResult = await workbench.call_tool("web_fetch", {"url": "https://example.com"})
        print(f"Tool Result: {result.result[0].content} (Name: {result.name})")
        return result
    except Exception as e:
        print(f"Error calling tool: {e}")
        return None
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the call tool with override example."""
    result = await call_tool_with_override()
    if not result:
        print("No result returned from tool execution.")

if __name__ == "__main__":
    asyncio.run(main())