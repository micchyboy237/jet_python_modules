import asyncio
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import AsyncIterator, List
from contextlib import asynccontextmanager
from mcp.server.fastmcp.server import FastMCP, Context
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(app: FastMCP[None]) -> AsyncIterator[None]:
    print("Starting FastMCP server...")
    yield
    print("Shutting down FastMCP server...")


server = FastMCP(
    name="FastMCPStandalone",
    instructions="A standalone MCP server with a sum tool.",
    debug=True,
    log_level="DEBUG",
    lifespan=lifespan
)


class SumInput(BaseModel):
    numbers: List[float] = Field(...,
                                 description="List of numbers to sum", min_items=1)


class SumOutput(BaseModel):
    result: float = Field(..., description="Sum of the input numbers")


@server.tool(description="Calculate the sum of a list of numbers.", annotations={"audience": ["user"], "priority": 0.8})
async def sum_numbers(arguments: SumInput, ctx: Context) -> SumOutput:
    await ctx.info(f"Calculating sum of {len(arguments.numbers)} numbers")
    try:
        total = sum(arguments.numbers)
        await ctx.report_progress(100, 100, "Sum calculated successfully")
        return SumOutput(result=total)
    except Exception as e:
        await ctx.error(f"Error calculating sum: {str(e)}")
        return SumOutput(result=0.0)


if __name__ == "__main__":
    server.run(transport="stdio")
