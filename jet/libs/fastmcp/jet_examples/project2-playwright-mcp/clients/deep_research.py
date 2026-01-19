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
