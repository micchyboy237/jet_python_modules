"""Simple browser navigation and content extraction demo using playwright-mcp."""

import asyncio
import yaml
from rich.console import Console

from fastmcp import Client
from utils.args import parse_common_args


console = Console()


async def main():
    args = parse_common_args("Simple Playwright-MCP browser demo")

    console.print("[bold green]Playwright-MCP Simple Browser Demo[/bold green]")
    console.print("[dim]Configuration used:[/dim]")
    console.print(f"  • Config   : {args.config}")
    console.print(f"  • URL      : {args.url}")
    console.print(f"  • Headless : {args.headless}")
    console.print(f"  • Timeout  : {args.timeout} ms\n")

    # Load multi-server config explicitly (more reliable across versions)
    with open(args.config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    client = Client(config_dict)

    async with client:
        try:
            nav_result = await client.call_tool(
                "browser_navigate",
                {"url": args.url}
            )
            console.print(f"[cyan]Navigated to:[/] {nav_result.data.get('url')}")
            console.print(f"[cyan]Page title:[/] {nav_result.data.get('title', '—')}")

            content_result = await client.call_tool(
                "browser_snapshot",
                {}   # many implementations don't need params; some accept timeout, etc.
            )
            # Different implementations return different structures
            if isinstance(content_result.data, dict):
                text = content_result.data.get("content", "") or content_result.data.get("text", "")
                title = content_result.data.get("title", "—")
            else:
                text = str(content_result.data)
                title = "—"

            preview = (text[:500] + "...") if len(text) > 500 else text

            console.print("\n[bold]Page content preview (first ~500 chars):[/bold]")
            console.print(preview)

        except Exception as e:
            console.print(f"[red]Error during demo:[/] {e}")


if __name__ == "__main__":
    asyncio.run(main())