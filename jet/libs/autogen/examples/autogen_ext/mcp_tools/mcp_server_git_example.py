#!/usr/bin/env python3
"""Example demonstrating retrieving git log using mcp-server-git."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import mcp_server_tools, StdioServerParams
import os

async def get_git_log(repo_path: str) -> List[Any]:
    """
    Retrieve git log using mcp-server-git tool.

    Args:
        repo_path (str): Path to the git repository.

    Returns:
        List[Any]: Tool execution result.
    """
    server_params = StdioServerParams(
        command="uvx",
        args=["mcp-server-git"],
        read_timeout_seconds=60,
    )
    
    try:
        tools = await mcp_server_tools(server_params=server_params)
        git_log_tool = next((tool for tool in tools if tool.name == "git_log"), None)
        if not git_log_tool:
            print("Git log tool not found.")
            return []
        
        result = await git_log_tool.run_json(
            args={"repo_path": repo_path},
            cancellation_token=CancellationToken(),
        )
        print(f"Git Log for {repo_path}:")
        for content in result:
            print(f"Content: {content.get('text', '')}")
        return result
    except Exception as e:
        print(f"Error retrieving git log: {e}")
        return []

async def main() -> None:
    """Main function to run the git log example."""
    repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    result = await get_git_log(repo_path=repo_path)
    if not result:
        print("No result returned from git log tool.")

if __name__ == "__main__":
    asyncio.run(main())