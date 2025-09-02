#!/usr/bin/env python3
"""Example demonstrating serialization and deserialization with tool overrides using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_core.tools import ToolOverride
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def serialize_and_load_with_overrides() -> Optional[McpWorkbench]:
    """
    Serialize and deserialize McpWorkbench with tool overrides.

    Returns:
        Optional[McpWorkbench]: Deserialized workbench instance or None if failed.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    overrides = {
        "fetch": ToolOverride(name="web_fetch", description="Enhanced web fetching tool"),
    }
    workbench = McpWorkbench(server_params=server_params, tool_overrides=overrides)
    
    try:
        config = workbench.dump_component()
        print(f"Serialized Config: {config.config}")
        loaded_workbench = McpWorkbench.load_component(config)
        print(f"Loaded Workbench Override: {loaded_workbench._tool_overrides['fetch'].name}")
        return loaded_workbench
    except Exception as e:
        print(f"Error serializing/loading config: {e}")
        return None
    finally:
        await workbench.stop()

async def main() -> None:
    """Main function to run the serialization with overrides example."""
    loaded_workbench = await serialize_and_load_with_overrides()
    if not loaded_workbench:
        print("No workbench loaded from config.")
    else:
        await loaded_workbench.stop()

if __name__ == "__main__":
    asyncio.run(main())