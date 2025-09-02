#!/usr/bin/env python3
"""Example demonstrating initializing an McpSessionActor with a real MCP server."""
import asyncio
from pathlib import Path
from typing import Optional
from autogen_ext.tools.mcp import McpSessionActor, StdioServerParams

async def initialize_actor() -> Optional[McpSessionActor]:
    """
    Initialize an McpSessionActor with a real MCP server.

    Returns:
        Optional[McpSessionActor]: Initialized actor or None if failed.
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
        print(f"Actor initialized: {actor.name}, active: {actor._active}")
        return actor
    except Exception as e:
        print(f"Error initializing actor: {e}")
        return None
    finally:
        if 'actor' in locals():
            await actor.close()

async def main() -> None:
    """Main function to run the initialize actor example."""
    actor = await initialize_actor()
    if not actor:
        print("Failed to initialize actor.")

if __name__ == "__main__":
    asyncio.run(main())