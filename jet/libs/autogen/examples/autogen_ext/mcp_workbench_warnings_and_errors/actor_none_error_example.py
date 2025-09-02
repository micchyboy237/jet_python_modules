#!/usr/bin/env python3
"""Example demonstrating error handling when McpWorkbench actor fails to initialize."""
import asyncio
from typing import List
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def list_tools_with_failed_actor() -> List[dict]:
    """
    Attempt to list tools when actor initialization fails.

    Returns:
        List[dict]: Empty list if actor initialization fails.
    """
    # Use an invalid command to simulate server connection failure
    server_params = StdioServerParams(command="invalid_mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        tools = await workbench.list_tools()
        print("Available Tools:", [tool["name"] for tool in tools])
        return tools
    except RuntimeError as e:
        print(f"Error listing tools: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the actor none error example."""
    tools = await list_tools_with_failed_actor()
    if not tools:
        print("No tools returned due to actor initialization failure.")

if __name__ == "__main__":
    asyncio.run(main())