#!/usr/bin/env python3
"""Example demonstrating retrieving GitHub file contents using mcp-server-github."""
import asyncio
from typing import Any, List
from autogen_core import CancellationToken
from autogen_ext.tools.mcp import mcp_server_tools, StdioServerParams
import os

async def get_github_file_contents(owner: str, repo: str, path: str, branch: str) -> List[Any]:
    """
    Retrieve file contents from a GitHub repository using mcp-server-github tool.

    Args:
        owner (str): GitHub repository owner.
        repo (str): Repository name.
        path (str): File path in the repository.
        branch (str): Branch name.

    Returns:
        List[Any]: Tool execution result.
    """
    github_token = os.environ.get("GITHUB_TOKEN")
    if not github_token:
        print("GITHUB_TOKEN environment variable is not set.")
        return []
    
    server_params = StdioServerParams(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_PERSONAL_ACCESS_TOKEN": github_token},
        read_timeout_seconds=60,
    )
    
    try:
        tools = await mcp_server_tools(server_params=server_params)
        file_tool = next((tool for tool in tools if tool.name == "get_file_contents"), None)
        if not file_tool:
            print("Get file contents tool not found.")
            return []
        
        result = await file_tool.run_json(
            args={"owner": owner, "repo": repo, "path": path, "branch": branch},
            cancellation_token=CancellationToken(),
        )
        print(f"GitHub File Contents ({owner}/{repo}/{path}):")
        for content in result:
            print(f"Content: {content.get('text', '')}")
        return result
    except Exception as e:
        print(f"Error retrieving GitHub file contents: {e}")
        return []

async def main() -> None:
    """Main function to run the GitHub file contents example."""
    result = await get_github_file_contents(
        owner="microsoft",
        repo="autogen",
        path="README.md",
        branch="main",
    )
    if not result:
        print("No result returned from GitHub file contents tool.")

if __name__ == "__main__":
    asyncio.run(main())