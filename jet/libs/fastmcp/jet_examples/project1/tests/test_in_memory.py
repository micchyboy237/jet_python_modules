"""In-memory FastMCP server tests."""
import pytest
from fastmcp import FastMCP, Client
from dataclasses import dataclass

@dataclass
class MockResult:
    value: str

@pytest.mark.asyncio
async def test_in_memory_tool():
    # Given
    mcp = FastMCP("TestServer")
    @mcp.tool
    def echo(text: str) -> str:
        return text.upper()
    client = Client(mcp)
    # When
    async with client:
        result = await client.call_tool("echo", {"text": "hello world"})
    # Then
    expected = "HELLO WORLD"
    assert result.data == expected
