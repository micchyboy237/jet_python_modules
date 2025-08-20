# calculator_tool.py
# Demonstrates logging tool invocations using Context.request_id
# Logs inputs and outputs for debugging purposes

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession
import logging

# Configure logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize FastMCP instance for calculator tool
mcp = FastMCP(name="Calculator")

# Tool to add two numbers with request logging


@mcp.tool()
async def sum(a: int, b: int, ctx: Context[ServerSession, None]) -> int:
    """Add two numbers together.

    Args:
        a: First integer
        b: Second integer
        ctx: Context object for session and request tracking

    Returns:
        Sum of the two numbers
    """
    logging.info(f"Request {ctx.request_id}: sum({a}, {b})")
    result = a + b
    logging.info(f"Request {ctx.request_id}: result = {result}")
    return result
