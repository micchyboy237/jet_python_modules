"""Basic connection & tool discovery tests for playwright-mcp."""

import pytest
from pathlib import Path
from fastmcp import Client

pytestmark = pytest.mark.asyncio

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # utils/ → project2-playwright-mcp/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mcp-config.yaml"


async def test_can_connect_and_list_tools():
    # Given
    client = Client(str(PROJECT_ROOT))

    # When
    async with client:
        tools = await client.list_tools()

    # Then
    tool_names = {t.name for t in tools}
    expected_common = {
        "playwright:navigate",
        "playwright:get_page_content",
        # Add more you expect from your playwright-mcp version
    }
    assert len(tool_names) >= 3, "Should discover at least a few tools"
    assert "playwright:navigate" in tool_names, "navigate tool missing"
