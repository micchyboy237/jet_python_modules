#!/usr/bin/env python3
"""Example demonstrating configuration serialization with McpWorkbench."""
import asyncio
from typing import Optional
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def serialize_and_load_config() -> Optional[McpWorkbench]:
    """
    Serialize and load McpWorkbench configuration.

    Returns:
        Optional[McpWorkbench]: Loaded workbench instance or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    workbench = McpWorkbench(server_params=server_params)
    
    try:
        config = workbench._to_config()
        print(f"Serialized Config: {config}")
        loaded_workbench = McpWorkbench._from_config(config)
        print(f"Loaded Workbench Server Command: {loaded_workbench.server_params.command}")
        return loaded_workbench
    except Exception as e:
        print(f"Error serializing/loading config: {e}")
        return None

async def main() -> None:
    """Main function to run the config serialization example."""
    loaded_workbench = await serialize_and_load_config()
    if not loaded_workbench:
        print("No workbench loaded from config.")

if __name__ == "__main__":
    asyncio.run(main())