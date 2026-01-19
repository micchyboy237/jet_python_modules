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
