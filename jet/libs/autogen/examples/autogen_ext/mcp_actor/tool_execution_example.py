#!/usr/bin/env python3
"""Example demonstrating tool execution using McpSessionActor."""
import asyncio
from pathlib import Path
from typing import Any, List
from autogen_ext.tools.mcp import McpSessionActor, StdioServerParams
from mcp.types import CallToolResult, ListToolsResult

async def execute_tools() -> List[Any]:
    """
    List and execute tools (echo and get_time) using McpSessionActor.

    Returns:
        List[Any]: Results of tool executions.
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
        tools_future = await actor.call("list_tools")
        tools_result: ListToolsResult = await tools_future
        print("Available Tools:", [tool.name for tool in tools_result.tools])
        
        results = []
        # Execute 'echo' tool
        echo_future = await actor.call("call_tool", {"name": "echo", "kargs": {"text": "Hello World"}})
        echo_result: CallToolResult = await echo_future
        if not echo_result.isError:
            results.append(echo_result.content)
            print(f"Echo Result: {echo_result.content[0].text}")
        
        # Execute 'get_time' tool
        time_future = await actor.call("call_tool", {"name": "get_time", "kargs": {}})
        time_result: CallToolResult = await time_future
        if not time_result.isError:
            results.append(time_result.content)
            print(f"Time Result: {time_result.content[0].text}")
        
        return results
    except Exception as e:
        print(f"Error executing tools: {e}")
        return []
    finally:
        await actor.close()

async def main() -> None:
    """Main function to run the tool execution example."""
    results = await execute_tools()
    if not results:
        print("No results returned from tool execution.")

if __name__ == "__main__":
    asyncio.run(main())