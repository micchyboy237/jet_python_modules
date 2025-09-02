#!/usr/bin/env python3
"""Example demonstrating tool override conflict detection using McpWorkbench."""
import asyncio
from typing import Optional
from autogen_core.tools import ToolOverride
from autogen_ext.tools.mcp import McpWorkbench, StdioServerParams

async def detect_override_conflict() -> Optional[McpWorkbench]:
    """
    Attempt to create McpWorkbench with conflicting tool overrides.

    Returns:
        Optional[McpWorkbench]: Workbench instance or None if conflict detected.
    """
    server_params = StdioServerParams(command="mcp_server", args=["--port", "8080"])
    overrides = {
        "fetch": ToolOverride(name="same_name"),
        "search": ToolOverride(name="same_name"),
    }
    
    try:
        workbench = McpWorkbench(server_params=server_params, tool_overrides=overrides)
        print("Workbench created successfully (unexpected).")
        return workbench
    except ValueError as e:
        print(f"Override conflict detected: {e}")
        return None

async def main() -> None:
    """Main function to run the override conflict detection example."""
    workbench = await detect_override_conflict()
    if not workbench:
        print("No workbench created due to override conflict.")
    else:
        await workbench.stop()

if __name__ == "__main__":
    asyncio.run(main())