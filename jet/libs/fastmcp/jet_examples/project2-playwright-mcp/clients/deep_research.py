"""Deep research pattern: follow links, extract content step-by-step."""

import argparse
import asyncio
import yaml
from typing import List, Dict, Any
from rich.console import Console

from fastmcp import Client
from utils.args import parse_common_args


console = Console()


def add_deep_research_args(parser: argparse.ArgumentParser) -> None:
    """Add script-specific arguments for deep research."""
    parser.add_argument(
        "--topic",
        "-t",
        required=True,
        help="Topic/phrase to search for in page content",
    )
    parser.add_argument(
        "--max-depth",
        "-d",
        type=int,
        default=3,
        help="Maximum crawl depth",
    )


async def perform_deep_research(
    client: Client,
    start_url: str,
    topic: str,
    max_depth: int,
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
            await client.call_tool("browser_navigate", {"url": url})
            content_result = await client.call_tool("browser_snapshot", {})
            if isinstance(content_result.data, dict):
                text = content_result.data.get("content", "") or content_result.data.get("text", "")
            else:
                text = str(content_result.data)

            if topic.lower() in text.lower():
                preview = text[:300].replace("\n", " ") + "..."
                console.print(f"[yellow]Relevant fragment found:[/] {preview}")

            # TODO: real link discovery – this is placeholder
            # Real version should parse snapshot/tree and find <a href=...>
            discovered_links = []  # ← implement later

            for next_url in discovered_links[:5]:  # limit per level
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
    args = parse_common_args(
        "Deep research crawler using Playwright-MCP",
        add_extra_args_callback=add_deep_research_args
    )

    console.print("[bold]Deep Research Parameters[/bold]")
    console.print(f"  • Config     : {args.config}")
    console.print(f"  • Start URL  : {args.url}")
    console.print(f"  • Topic      : {args.topic}")
    console.print(f"  • Max depth  : {args.max_depth}")
    console.print(f"  • Headless   : {args.headless}")
    console.print(f"  • Timeout    : {args.timeout} ms\n")

    # Load multi-server config explicitly (more reliable across versions)
    with open(args.config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    client = Client(config_dict)

    async with client:
        await perform_deep_research(
            client,
            start_url=args.url,
            topic=args.topic,
            max_depth=args.max_depth,
        )


if __name__ == "__main__":
    asyncio.run(main())