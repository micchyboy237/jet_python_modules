#!/usr/bin/env python3
"""Example demonstrating prompt operations using McpSessionActor."""
import asyncio
from pathlib import Path
from typing import Any, List
from autogen_ext.tools.mcp import McpSessionActor, StdioServerParams
from mcp.types import GetPromptResult, ListPromptsResult

async def execute_prompt_operations() -> List[Any]:
    """
    List and retrieve a code review prompt using McpSessionActor.

    Returns:
        List[Any]: Results of prompt operations.
    """
    server_path = Path(__file__).parent / "mcp_server_comprehensive.py"
    server_params = StdioServerParams(
        command="uv",
        args=["run", "python", str(server_path)],
        read_timeout_seconds=10,
    )
    actor = McpSessionActor(server_params=server_params)
    
    try:
        await actor.initialize()
        prompts_future = await actor.call("list_prompts")
        prompts_result: ListPromptsResult = await prompts_future
        print("Available Prompts:", [prompt.name for prompt in prompts_result.prompts])
        
        prompt_future = await actor.call(
            "get_prompt",
            {"name": "code_review", "kargs": {"arguments": {"code": "print('hello')", "language": "python"}}},
        )
        prompt_result: GetPromptResult = await prompt_future
        results = [prompt_result.messages]
        print(f"Code Review Prompt: {prompt_result.description}")
        print(f"Prompt Message: {prompt_result.messages[0].content.text}")
        return results
    except Exception as e:
        print(f"Error executing prompt operations: {e}")
        return []
    finally:
        await actor.close()

async def main() -> None:
    """Main function to run the prompt operations example."""
    results = await execute_prompt_operations()
    if not results:
        print("No results returned from prompt operations.")

if __name__ == "__main__":
    asyncio.run(main())