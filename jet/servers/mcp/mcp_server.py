import asyncio
from starlette.responses import JSONResponse
from starlette.requests import Request
from typing import Any, AsyncIterator
from contextlib import asynccontextmanager
from mcp.server.fastmcp.server import FastMCP
from jet.servers.mcp.tool_implementations import read_file, navigate_to_url, summarize_text, process_data
from jet.servers.mcp.models import FileInput, UrlInput, SummarizeTextInput


@asynccontextmanager
async def lifespan(app: FastMCP[None]) -> AsyncIterator[None]:
    print("Starting FastMCP server...")
    yield
    print("Shutting down FastMCP server...")

server = FastMCP(
    name="FastMCPStandalone",
    instructions="A standalone MCP server with file and browser tools.",
    debug=True,
    log_level="DEBUG",
    lifespan=lifespan
)


@server.tool(description="Read the contents of a file.", annotations={"audience": ["user"], "priority": 0.9})
async def read_file_tool(arguments: FileInput, ctx) -> Any:
    return await read_file(arguments, ctx)


@server.tool(description="Navigate to a URL and return the page title, links from the same server, and all visible text content.", annotations={"audience": ["assistant"], "priority": 0.8})
async def navigate_to_url_tool(arguments: UrlInput, ctx) -> Any:
    return await navigate_to_url(arguments, ctx)


@server.tool(description="Summarize text content to a specified word limit.", annotations={"audience": ["assistant"], "priority": 0.7})
async def summarize_text_tool(arguments: SummarizeTextInput, ctx) -> Any:
    return await summarize_text(arguments, ctx)


@server.tool(description="Process data with progress.", annotations={"audience": ["user"]})
async def process_data_tool(data: str, ctx) -> str:
    return await process_data(data, ctx)


@server.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


@server.resource("resource://welcome", description="A welcome message")
async def welcome_message() -> str:
    return "Welcome to FastMCP!"


@server.prompt(description="Analyze a file")
async def analyze_file(path: str) -> List[Dict]:
    content = open(path, "r").read()
    return [{"role": "user", "content": f"Analyze this content:\n{content}"}]

if __name__ == "__main__":
    server.run(transport="stdio")
