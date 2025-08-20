# jet_python_modules/jet/servers/mcp/tools_to_discover/codegen_tool.py
# Demonstrates multi-step workflow with session state
# Tracks progress in a code generation task using custom session

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
from typing import List

mcp = FastMCP(name="CodeGen")


@mcp.tool()
async def add_code_step(step: str, ctx: Context[ServerSession, None]) -> str:
    """Add a step to the code generation process.
    Args:
        step: Name of the step to add
        ctx: Context object with session for state tracking
    Returns:
        String with added step and current steps list
    """
    # Store steps in session state (e.g., a dictionary)
    if not hasattr(ctx.session, "steps"):
        ctx.session.steps = []
    ctx.session.steps.append(step)
    return f"Step '{step}' added. Current steps: {ctx.session.steps}"
