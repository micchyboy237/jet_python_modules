#!/usr/bin/env python3
"""Example demonstrating reading a file using mcp-server-filesystem."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import mcp_server_tools, StdioServerParams

async def read_filesystem_file(path: str) -> List[Any]:
    """
    Read a file using mcp-server-filesystem tool.

    Args:
        path (str): Path to the file to read.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        read_timeout_seconds=60,
    )
    
    try:
        tools = await mcp_server_tools(server_params=server_params)
        read_file_tool = next((tool for tool in tools if tool.name == "read_file"), None)
        if not read_file_tool:
            print("Read file tool not found.")
            return []
        
        result = await read_file_tool.run_json(
            args={"path": path},
            cancellation_token=CancellationToken(),
        )
        print(f"File Content ({path}):")
        for content in result:
            print(f"Content: {content.get('text', '')}")
        return result
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

async def main() -> None:
    """Main function to run the filesystem read example."""
    result = await read_filesystem_file(path="README.md")
    if not result:
        print("No result returned from read file tool.")

if __name__ == "__main__":
    asyncio.run(main())