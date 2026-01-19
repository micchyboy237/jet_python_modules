"""Stdio transport tests."""
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_stdio_connection():
    # Given
    client = Client("mcp_servers/financial_analyzer/server.py")
    # When / Then
    async with client:
        tools = await client.list_tools()
        expected_tool_names = ["get_stock_price", "calculate_portfolio_value"]
        assert [t.name for t in tools] == expected_tool_names
