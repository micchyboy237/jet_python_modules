#!/usr/bin/env python3
"""Example demonstrating resource operations using McpSessionActor."""
import asyncio
import json
from pathlib import Path
from typing import Any, List
from autogen_ext.tools.mcp import McpSessionActor, StdioServerParams
from mcp.types import ListResourcesResult, ReadResourceResult

async def execute_resource_operations() -> List[Any]:
    """
    List and read company user resources using McpSessionActor.

    Returns:
        List[Any]: Results of resource operations.
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
        resources_future = await actor.call("list_resources")
        resources_result: ListResourcesResult = await resources_future
        print("Available Resources:", [resource.name for resource in resources_result.resources])
        
        read_future = await actor.call("read_resource", {"name": None, "kargs": {"uri": "file:///company/users.json"}})
        read_result: ReadResourceResult = await read_future
        users_data = json.loads(read_result.contents[0].text)
        print(f"Users Data: {users_data}")
        return [users_data]
    except Exception as e:
        print(f"Error executing resource operations: {e}")
        return []
    finally:
        await actor.close()

async def main() -> None:
    """Main function to run the resource operations example."""
    results = await execute_resource_operations()
    if not results:
        print("No results returned from resource operations.")

if __name__ == "__main__":
    asyncio.run(main())