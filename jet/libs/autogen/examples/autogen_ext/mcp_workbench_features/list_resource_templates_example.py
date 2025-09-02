#!/usr/bin/env python3
"""Example demonstrating listing resource templates using McpWorkbench."""
import asyncio
from typing import List
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import ListResourceTemplatesResult

async def list_available_resource_templates() -> List[dict]:
    """
    List available resource templates using McpWorkbench with auto-start.

    Returns:
        List[dict]: List of resource template dictionaries.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        result: ListResourceTemplatesResult = await workbench.list_resource_templates()
        templates = [
            {"uriTemplate": template.uriTemplate, "name": template.name, "mimeType": template.mimeType}
            for template in result.resourceTemplates
        ]
        print("Available Resource Templates:")
        for template in templates:
            print(f"- {template['name']} ({template['uriTemplate']}): {template['mimeType']}")
        return templates
    except Exception as e:
        print(f"Error listing resource templates: {e}")
        return []
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the list resource templates example."""
    templates = await list_available_resource_templates()
    if not templates:
        print("No resource templates returned.")

if __name__ == "__main__":
    asyncio.run(main())