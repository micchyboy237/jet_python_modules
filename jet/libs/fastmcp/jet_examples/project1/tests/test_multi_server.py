"""Multi-server config tests."""
import pytest
from fastmcp import Client

@pytest.mark.asyncio
async def test_multi_server_tools():
    # Given
    client = Client("mcp-config.yaml")
    # When
    async with client:
        tools = await client.list_tools()
    # Then
    expected = ["financial_get_stock_price", "financial_calculate_portfolio_value", "rag_semantic_search"]
    actual = [t.name for t in tools]
    assert sorted(actual) == sorted(expected)
