#!/usr/bin/env python3
"""Example demonstrating fetching content using mcp-server-fetch."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import mcp_server_tools, StdioServerParams

async def fetch_web_content(url: str) -> List[Any]:
    """
    Fetch content from a URL using mcp-server-fetch tool.

    Args:
        url (str): URL to fetch content from.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = StdioServerParams(
        command="uvx",
        args=["mcp-server-fetch"],
        read_timeout_seconds=60,
    )
    
    try:
        tools = await mcp_server_tools(server_params=server_params)
        fetch_tool = next((tool for tool in tools if tool.name == "fetch"), None)
        if not fetch_tool:
            print("Fetch tool not found.")
            return []
        
        result = await fetch_tool.run_json(
            args={"url": url},
            cancellation_token=CancellationToken(),
        )
        print(f"Fetch Result for {url}:")
        for content in result:
            print(f"Content: {content.get('text', '')}")
        return result
    except Exception as e:
        print(f"Error fetching content: {e}")
        return []

async def main() -> None:
    """Main function to run the fetch tool example."""
    result = await fetch_web_content(url="https://github.com/")
    if not result:
        print("No result returned from fetch tool.")

if __name__ == "__main__":
    asyncio.run(main())