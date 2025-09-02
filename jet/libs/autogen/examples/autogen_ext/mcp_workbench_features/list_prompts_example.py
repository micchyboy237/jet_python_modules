#!/usr/bin/env python3
"""Example demonstrating listing prompts using McpWorkbench."""
import asyncio
from typing import List
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import ListPromptsResult

async def list_available_prompts() -> List[dict]:
    """
    List available prompts using McpWorkbench with auto-start.

    Returns:
        List[dict]: List of prompt dictionaries.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        result: ListPromptsResult = await workbench.list_prompts()
        prompts = [
            {"name": prompt.name, "description": prompt.description}
            for prompt in result.prompts
        ]
        print("Available Prompts:")
        for prompt in prompts:
            print(f"- {prompt['name']}: {prompt['description']}")
        return prompts
    except Exception as e:
        print(f"Error listing prompts: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the list prompts example."""
    prompts = await list_available_prompts()
    if not prompts:
        print("No prompts returned.")

if __name__ == "__main__":
    asyncio.run(main())