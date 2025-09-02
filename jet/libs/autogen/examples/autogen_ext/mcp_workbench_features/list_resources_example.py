#!/usr/bin/env python3
"""Example demonstrating listing resources using McpWorkbench."""
import asyncio
from typing import List
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import ListResourcesResult

async def list_available_resources() -> List[dict]:
    """
    List available resources using McpWorkbench with auto-start.

    Returns:
        List[dict]: List of resource dictionaries.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        result: ListResourcesResult = await workbench.list_resources()
        resources = [
            {"uri": str(resource.uri), "name": resource.name, "mimeType": resource.mimeType}
            for resource in result.resources
        ]
        print("Available Resources:")
        for resource in resources:
            print(f"- {resource['name']} ({resource['uri']}): {resource['mimeType']}")
        return resources
    except Exception as e:
        print(f"Error listing resources: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the list resources example."""
    resources = await list_available_resources()
    if not resources:
        print("No resources returned.")

if __name__ == "__main__":
    asyncio.run(main())