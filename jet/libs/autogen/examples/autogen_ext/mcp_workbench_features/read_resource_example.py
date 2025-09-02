#!/usr/bin/env python3
"""Example demonstrating reading a resource using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams
from mcp.types import ReadResourceResult

async def read_document_resource() -> Optional[ReadResourceResult]:
    """
    Read a document resource using McpWorkbench with auto-start.

    Returns:
        Optional[ReadResourceResult]: Resource read result or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        uri = "file:///test/document.txt"
        result: ReadResourceResult = await workbench.read_resource(uri)
        print(f"Resource Content ({uri}):")
        for content in result.contents:
            print(f"- {content.text} (MIME: {content.mimeType})")
        return result
    except Exception as e:
        print(f"Error reading resource: {e}")
        return None
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the read resource example."""
    result = await read_document_resource()
    if not result:
        print("No resource content returned.")

if __name__ == "__main__":
    asyncio.run(main())