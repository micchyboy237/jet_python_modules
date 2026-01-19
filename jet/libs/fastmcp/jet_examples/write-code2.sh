#!/usr/bin/env bash

# setup_project2_playwright_mcp.sh
# Unix shell script to create the project2-playwright-mcp folder structure
# and populate starter files for using the real playwright-mcp server
# with FastMCP Python client (HTTP mode recommended)
#
# Assumptions / Requirements:
#   - Node.js + npm installed (to run playwright-mcp server)
#   - Python 3.10+ environment ready
#   - Run this script from the parent directory where you want jet/libs/fastmcp/jet_examples/
#     to be created (or adjust PROJECT_PARENT below)
#
# Usage:
#   chmod +x setup_project2_playwright_mcp.sh
#   ./setup_project2_playwright_mcp.sh

set -euo pipefail

PROJECT_PARENT="/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/fastmcp/jet_examples/project2"
PROJECT_DIR="${PROJECT_PARENT}/project2-playwright-mcp"

echo "Creating Playwright-MCP example project in:"
echo "  ${PROJECT_DIR}"
echo ""

mkdir -p "${PROJECT_DIR}"/{clients,tests}

cd "${PROJECT_PARENT}" || { echo "Cannot cd to ${PROJECT_PARENT} — create parent dirs first"; exit 1; }

# ------------------------------------------------------------------
# requirements.txt
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/requirements.txt" << 'EOF'
fastmcp>=2.14.0
rich>=13.7
python-dotenv>=1.0
pytest>=8.0
pytest-asyncio>=0.23
EOF

# ------------------------------------------------------------------
# .env
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/.env" << 'EOF'
# playwright-mcp server location (HTTP mode — recommended)
PLAYWRIGHT_MCP_URL=http://localhost:8931

# Optional browser behavior (passed as env to server if using stdio mode)
BROWSER_HEADLESS=true
DEFAULT_TIMEOUT_MS=45000
EOF

# ------------------------------------------------------------------
# mcp-config.yaml  (HTTP transport — simplest & most reliable with Python client)
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/mcp-config.yaml" << 'EOF'
mcpServers:
  playwright:
    transport: http
    url: ${PLAYWRIGHT_MCP_URL}
    # Optional: if your instance has auth
    # headers:
    #   Authorization: Bearer your-token-here
EOF

# ------------------------------------------------------------------
# clients/simple_browser.py
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/clients/simple_browser.py" << 'EOF'
"""Simple browser navigation and content extraction demo using playwright-mcp."""

import asyncio
import os
from rich.console import Console
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()

console = Console()


async def main():
    client = Client("mcp-config.yaml")

    async with client:
        console.print("[bold green]Playwright-MCP Simple Browser Demo[/bold green]")

        try:
            # 1. Navigate to a page
            nav_result = await client.call_tool(
                "playwright:navigate",
                {"url": "https://news.ycombinator.com"}
            )
            console.print(f"[cyan]Navigated to:[/] {nav_result.data.get('url')}")
            console.print(f"[cyan]Page title:[/] {nav_result.data.get('title', '—')}")

            # 2. Extract main content (accessibility-tree based, usually clean text)
            content_result = await client.call_tool(
                "playwright:get_page_content",
                {"max_length": 3000}
            )
            text = content_result.data
            preview = text[:500] + "..." if len(text) > 500 else text

            console.print("\n[bold]Page content preview (first ~500 chars):[/bold]")
            console.print(preview)

        except Exception as e:
            console.print(f"[red]Error during demo:[/] {e}")


if __name__ == "__main__":
    asyncio.run(main())
EOF

# ------------------------------------------------------------------
# clients/deep_research.py
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/clients/deep_research.py" << 'EOF'
"""Deep research pattern: follow links, extract content step-by-step."""

import asyncio
from typing import List, Dict, Any
from rich.console import Console
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()

console = Console()


async def perform_deep_research(
    client: Client,
    start_url: str,
    topic: str,
    max_depth: int = 3
):
    """Naive breadth-first crawl looking for topic-relevant content."""
    visited = set()
    queue: List[Dict[str, Any]] = [{"url": start_url, "depth": 0, "context": ""}]

    console.print(f"[bold green]Starting deep research on:[/] {topic!r}")
    console.print(f"[dim]Start:[/] {start_url}  (max depth: {max_depth})[/dim]\n")

    while queue:
        current = queue.pop(0)
        url = current["url"]
        depth = current["depth"]

        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        console.print(f"[cyan]Depth {depth} → Visiting:[/] {url}")

        try:
            # Navigate
            await client.call_tool("playwright:navigate", {"url": url})

            # Get structured content
            content_result = await client.call_tool(
                "playwright:get_page_content",
                {"max_length": 6000}
            )
            text: str = content_result.data

            # Very naive relevance check — replace with LLM scoring in production
            if topic.lower() in text.lower():
                preview = text[:300].replace("\n", " ") + "..."
                console.print(f"[yellow]Relevant fragment found:[/] {preview}")

            # Placeholder: discover next links
            # Real version: call playwright:get_links or parse <a href> via tool
            # Here we just simulate continuation
            next_urls = [url.rstrip("/") + f"/page{depth+2}"]  # dummy

            for next_url in next_urls:
                if next_url not in visited:
                    queue.append({
                        "url": next_url,
                        "depth": depth + 1,
                        "context": text[:500]
                    })

        except Exception as e:
            console.print(f"[red]Error visiting {url}:[/] {e}")

    console.print("\n[bold green]Research finished.[/] Visited {} pages.".format(len(visited)))


async def main():
    client = Client("mcp-config.yaml")
    async with client:
        await perform_deep_research(
            client,
            start_url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            topic="neural networks history",
            max_depth=2
        )


if __name__ == "__main__":
    asyncio.run(main())
EOF

# ------------------------------------------------------------------
# tests/test_browser_connection.py
# ------------------------------------------------------------------
cat > "${PROJECT_DIR}/tests/test_browser_connection.py" << 'EOF'
"""Basic connection & tool discovery tests for playwright-mcp."""

import pytest
from fastmcp import Client

pytestmark = pytest.mark.asyncio


async def test_can_connect_and_list_tools():
    # Given
    client = Client("mcp-config.yaml")

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
EOF
