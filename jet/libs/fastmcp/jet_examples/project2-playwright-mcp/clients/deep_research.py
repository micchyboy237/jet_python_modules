"""Deep research crawler: starts from SearXNG search, follows relevant links."""

import argparse
import asyncio
import json
import re
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse, urlencode

import httpx
import yaml
from rich.console import Console
from fastmcp import Client
from utils.args import parse_common_args

console = Console()


def extract_links_naive(base_url: str, text: str) -> List[str]:
    """Simple regex-based link extraction (fallback when LLM is not used)."""
    patterns = [
        r'\[([^\]]+)\]\((https?://[^)]+)\)',
        r'(https?://[^\s<>"\'\)]+)',
        r'href=["\'](.*?)["\']',
    ]
    found = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            url = match.group(2) if len(match.groups()) > 1 else match.group(1)
            if not url:
                continue
            if url.startswith('/'):
                url = urljoin(base_url, url)
            if url.startswith(('http://', 'https://')):
                found.add(url)
    return list(found)


async def extract_links_with_llm(
    content: str,
    current_url: str,
    query: str,
    llm_url: str = "http://shawn-pc.local:8080/v1",
    model: str = "ggml-model-q4_0.gguf",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Use local LLM to summarize page and extract promising follow-up links."""
    client = httpx.AsyncClient(timeout=90.0)
    truncated = content[:12000] + "..." if len(content) > 12000 else content

    prompt = f"""You are a focused web researcher.
Current query: {query}
Current page: {current_url}

Page content excerpt:
{truncated}

Tasks:
1. Write very concise summary (2-4 sentences).
2. Extract 3-8 most promising follow-up URLs likely relevant to the query.
   - Prefer article/content pages over navigation, ads, pagination
   - Prefer same or closely related domain
   - Give each a relevance score 0-10 and short reason

Output strictly JSON:
{{
  "summary": "short summary here",
  "candidates": [
    {{"url": "https://...", "relevance": 8, "reason": "short reason"}},
    ...
  ]
}}
"""

    try:
        resp = await client.post(
            f"{llm_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25,
                "max_tokens": 600,
            },
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"].strip()

        if raw.startswith("```json"):
            raw = raw.split("```json", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)
        return data.get("summary", "No summary"), data.get("candidates", [])
    except Exception as e:
        console.print(f"[red]LLM extraction failed:[/] {str(e)[:180]}")
        return "LLM call failed", []


async def perform_deep_research(
    client: Client,
    start_url: str,
    query: str,
    max_depth: int = 3,
    use_llm: bool = False,
    llm_base_url: str = "http://shawn-pc.local:8080/v1",
    same_domain_only: bool = True,
):
    visited: set[str] = set()
    start_domain = urlparse(start_url).netloc

    queue: List[Dict[str, Any]] = [{
        "url": start_url,
        "depth": 0,
        "relevance": 10.0,
        "title": "Initial search"
    }]

    console.print(f"[bold green]Deep research on:[/] {query!r}   (max depth: {max_depth})")
    console.print(f"[dim]Mode:[/] {'LLM-powered' if use_llm else 'Simple regex'} link discovery")
    if same_domain_only:
        console.print(f"[dim]Domain restriction:[/] {start_domain}\n")

    while queue:
        item = queue.pop(0)
        url = item["url"]
        depth = item["depth"]

        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        console.rule(f"[cyan]Depth {depth} • {url}[/cyan]")

        try:
            await client.call_tool("browser_navigate", {"url": url})

            # Option 1 — Recommended: wait for characteristic result text (best for SearXNG)
            try:
                await client.call_tool("browser_wait_for", {
                    "text": "results",           # or "SearXNG", "Playwright", "Next page", etc.
                    "timeout": 30000,             # many implementations accept timeout separately
                    # "time": 10, # Wait up to 10 seconds
                })
                console.print("[green]✓ Waited for search results text to appear[/green]")
            except Exception as e:
                console.print(f"[yellow]Text wait failed ({str(e)[:80]}), falling back...[/yellow]")

            # Short settle time + optional network idle wait
            await asyncio.sleep(1.5)
            try:
                await client.call_tool("browser_wait_for", {
                    "event": "networkidle",      # ← this is also NOT supported in official
                    "timeout": 12000,
                    # "time": 10, # Wait up to 10 seconds
                })
            except:
                pass

            result = await client.call_tool("browser_snapshot", {})
            data = result.data if isinstance(result.data, dict) else {}
            text = data.get("text", "") or ""
            title = data.get("title", "—") or "—"

            # ── Fallback if accessibility tree is empty/short ──────────────
            if not text.strip() or len(text.strip()) < 400:
                try:
                    body_text_result = await client.call_tool("browser_evaluate", {
                        "script": "document.body.innerText"
                    })
                    console.print("[yellow]Used browser_evaluate → innerText[/yellow]")
                except Exception as e1:
                    console.print(f"[dim]innerText evaluate failed: {str(e1)[:120]}[/dim]")
                    try:
                        body_text_result = await client.call_tool("browser_evaluate", {
                            "script": "document.body.textContent"
                        })
                        console.print("[yellow]Used browser_evaluate → textContent fallback[/yellow]")
                    except Exception as e2:
                        console.print(f"[dim]textContent evaluate failed: {str(e2)[:120]}[/dim]")
                        body_text_result = None

                if body_text_result and hasattr(body_text_result, "data") and body_text_result.data:
                    text = str(body_text_result.data).strip()

            # Log more info about what we actually got
            console.print(f"[dim]Content length after extraction:[/] {len(text)} chars")

            if any(kw in text.lower() for kw in ["captcha", "prove you're not", "challenge"]):
                console.print("[red bold]CAPTCHA/CHALLENGE DETECTED — skipping[/red bold]")
                continue

            console.print(f"[bold]Title:[/] {title}")
            preview = text[:360].replace('\n', ' ').strip()
            console.print(f"[dim]Preview (first 360 chars / {len(text)} total):[/dim] {preview}...\n")

            # ── Discover next pages ────────────────────────────────
            if use_llm:
                summary, candidates = await extract_links_with_llm(text, url, query, llm_base_url)
                console.print(f"[cyan]Summary:[/] {summary[:220]}{'...' if len(summary)>220 else ''}\n")
                next_items = [
                    {
                        "url": c["url"],
                        "depth": depth + 1,
                        "relevance": float(c.get("relevance", 5)),
                        "title": c.get("reason", "(LLM)")
                    }
                    for c in candidates if c.get("url")
                ]
            else:
                raw_links = extract_links_naive(url, text)
                next_items = [
                    {
                        "url": link,
                        "depth": depth + 1,
                        "relevance": 7.5 if query.lower() in link.lower() else 3.5,
                        "title": "(regex)"
                    }
                    for link in raw_links
                ]

            # Filtering & sorting
            if same_domain_only:
                next_items = [i for i in next_items if urlparse(i["url"]).netloc == start_domain]
            next_items = [i for i in next_items if i["url"] not in visited]
            next_items.sort(key=lambda x: x["relevance"], reverse=True)
            to_queue = next_items[:5]

            queue.extend(to_queue)
            console.print(f"[dim]→ {len(next_items)} candidates → queued {len(to_queue)}[/dim]")

        except Exception as e:
            console.print(f"[red]Error visiting {url}:[/] {str(e)[:240]}...")
            continue

    console.print(f"\n[bold green]Finished[/] • Visited {len(visited)} pages")



def add_deep_research_args(parser: argparse.ArgumentParser) -> None:
    """Add script-specific arguments"""
    parser.add_argument(
        "query",
        nargs="?",
        help="Search query / topic (positional or via --query)"
    )
    parser.add_argument(
        "--query", "-q",
        help="Search query / topic",
        dest="query_opt"  # avoid conflict with positional
    )
    parser.add_argument(
        "--max-depth", "-d",
        type=int,
        default=3,
        help="Maximum crawl depth"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use local LLM for smarter link discovery & scoring"
    )
    parser.add_argument(
        "--llm-url",
        default="http://shawn-pc.local:8080/v1",
        help="Local OpenAI-compatible LLM endpoint"
    )
    parser.add_argument(
        "--same-domain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only follow links from same domain as start URL"
    )


async def main():
    args = parse_common_args(
        "Deep research crawler – SearXNG → relevant pages",
        add_extra_args_callback=add_deep_research_args
    )

    # Determine final query value (positional > --query > error)
    query = args.query or args.query_opt
    if not query:
        console.print("[red]Error:[/] Please provide a query (positional argument or --query / -q)")
        return

    # Better URL construction + possibility to add future params easily
    search_params = {
        "q": query.strip(),
        # "format": "html",          # optional
        # "categories": "general",
    }
    query_string = urlencode(search_params)
    start_url =  f"http://jethros-macbook-air.local:8888/search?{query_string}"
    console.print(f"[dim italic]Auto-generated starting URL:[/] {start_url}\n")

    console.print("[bold]Parameters[/bold]")
    console.print(f"  • Query        : {query}")
    console.print(f"  • Start URL    : {start_url}")
    console.print(f"  • Max depth    : {args.max_depth}")
    console.print(f"  • Use LLM      : {args.use_llm}")
    console.print(f"  • Same domain  : {args.same_domain}\n")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = Client(config)
    async with client:
        await perform_deep_research(
            client=client,
            start_url=start_url,
            query=query,
            max_depth=args.max_depth,
            use_llm=args.use_llm,
            llm_base_url=args.llm_url,
            same_domain_only=args.same_domain,
        )


if __name__ == "__main__":
    asyncio.run(main())