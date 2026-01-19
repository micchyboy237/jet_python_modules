
"""Improved demo: SearXNG search → click first result → read article"""

import asyncio
import yaml
from rich.console import Console
from fastmcp import Client
from utils.args import parse_common_args

console = Console()

async def main():
    args = parse_common_args("Playwright-MCP: Search → First Result → Article")

    console.print("[bold green]Improved Playwright-MCP Demo[/bold green]")
    console.print("Flow: SearXNG search → wait for results → click first link → wait for content → snapshot\n")

    console.print("[dim]Configuration used:[/dim]")
    console.print(f"  • Config   : {args.config}")
    console.print(f"  • URL      : {args.url}")
    console.print(f"  • Headless : {args.headless}")
    console.print(f"  • Timeout  : {args.timeout} ms\n")

    with open(args.config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    client = Client(config_dict)
    async with client:
        try:
            # 1. Go to search page
            await client.call_tool("browser_navigate", {"url": args.url})
            console.print(f"[cyan]Navigated to search:[/] {args.url}")

            # 2. Wait for search results to appear
            await client.call_tool("browser_wait_for", {
                "selector": "css=.results article, css=.result",
                "state": "visible",
                "timeout": args.timeout
            })
            console.print("[green]✓ Search results loaded[/green]")

            # 3. Click the first meaningful result link
            # Common robust selectors for SearXNG results
            first_result_selector = "css=.results article:first-child a.title, css=.result:first-child a[href^='http']"

            await client.call_tool("browser_click", {
                "selector": first_result_selector,
                "timeout": 15000
            })
            console.print("[green]✓ Clicked first result link[/green]")

            # 4. Wait for article/main content to load
            await client.call_tool("browser_wait_for", {
                "selector": "css=article, main, .content, .post-content, css=h1",
                "state": "visible",
                "timeout": args.timeout
            })
            console.print("[green]✓ Article/main content loaded[/green]")

            # 5. Optional: take screenshot for debugging
            screenshot_result = await client.call_tool("browser_take_screenshot", {"full_page": True})
            if screenshot_result.data:
                console.print("[dim]Screenshot taken (full page)[/dim]")

            # 6. Get final content
            content_result = await client.call_tool("browser_snapshot", {})
            if isinstance(content_result.data, dict):
                text = content_result.data.get("content", "") or content_result.data.get("text", "")
                title = content_result.data.get("title", "—")
            else:
                text = str(content_result.data)
                title = "—"

            preview = (text[:800] + "...") if len(text) > 800 else text
            console.print(f"\n[bold]Final page title:[/] {title}")
            console.print("\n[bold]Article/content preview (first ~800 chars):[/bold]")
            console.print(preview)

        except Exception as e:
            console.print(f"[red]Error during demo:[/] {e}")
            console.print("[yellow]Tip:[/] Try different selectors if the site layout changed.")
            console.print("      Current selectors: .results article, .result:first-child a[href^='http']")

if __name__ == "__main__":
    asyncio.run(main())