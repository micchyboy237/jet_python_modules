#!/usr/bin/env python3
"""Example demonstrating getting a prompt with arguments using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import GetPromptResult

async def get_python_code_review_prompt() -> Optional[GetPromptResult]:
    """
    Get the 'code_review' prompt with Python-specific arguments using McpWorkbench.

    Returns:
        Optional[GetPromptResult]: Prompt result or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        arguments = {"language": "python", "style": "pep8"}
        result: GetPromptResult = await workbench.get_prompt("code_review", arguments=arguments)
        print(f"Prompt Description: {result.description}")
        print(f"Prompt Message: {result.messages[0].content.text}")
        return result
    except Exception as e:
        print(f"Error getting prompt: {e}")
        return None
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the get prompt with arguments example."""
    result = await get_python_code_review_prompt()
    if not result:
        print("No prompt returned.")

if __name__ == "__main__":
    asyncio.run(main())