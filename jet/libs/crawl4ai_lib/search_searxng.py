import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional, TypedDict
from urllib.parse import urlencode, urljoin

import httpx
import numpy as np
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


# ============================================================
# Type Definitions
# ============================================================


class SemanticResult(TypedDict):
    """Typed dictionary for semantic search/reranking results."""

    rank: int
    score: float
    title: str
    url: str
    snippet: str


# ============================================================
# CLI + Config
# ============================================================


@dataclass
class AppConfig:
    query: str
    top_k: int
    sites: Optional[List[str]]


# ============================================================
# Embedding & Similarity
# ============================================================


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def embed_texts(
    texts: List[str],
    base_url: str = os.getenv("LLAMA_CPP_EMBED_URL"),
    embed_model: str = os.getenv("LLAMA_CPP_EMBED_MODEL"),
) -> List[np.ndarray]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/embeddings",
            json={"model": embed_model, "input": texts},
        )
        response.raise_for_status()

        data = response.json()
        return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]


# ============================================================
# Semantic Reranking
# ============================================================


async def semantic_seed_filter(
    query: str,
    results: List[dict],
    top_k: int = 8,
    embed_url: str = os.getenv("LLAMA_CPP_EMBED_URL"),
) -> List[SemanticResult]:
    """
    Perform semantic reranking and return results with rank, score, title, url, and snippet.
    """
    if not results:
        return []

    console.rule("Semantic Reranking", style="bright_blue")

    task_description = "[cyan]Computing embeddings..."
    with Live(console=console, refresh_per_second=8) as live:
        live.update(Panel(task_description, style="bold cyan"))

        texts = [query] + [
            f"{r.get('title', '')} {r.get('snippet', '')}" for r in results
        ]
        embeddings = await embed_texts(texts, base_url=embed_url)

        live.update(
            Panel(
                "[green]Embeddings ready — calculating similarities...",
                style="bold green",
            )
        )

        # Calculate similarity scores
        scored: List[SemanticResult] = []
        for i, (r, emb) in enumerate(zip(results, embeddings[1:]), 1):
            score = cosine_similarity(embeddings[0], emb)

            scored.append(
                {
                    "rank": i,
                    "score": score,
                    "title": r.get("title", "").strip(),
                    "url": r.get("url", "").strip(),
                    "snippet": r.get("snippet", "").strip(),
                }
            )

        # Sort by similarity score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # Update ranks to final order
        for i, item in enumerate(scored, 1):
            item["rank"] = i

        live.update(
            Panel("[green]Similarity scores calculated and sorted", style="bold green")
        )

    await asyncio.sleep(0.6)
    return scored[:top_k]


# ============================================================
# Site Normalization (unchanged)
# ============================================================


def normalize_sites(raw_sites: Optional[List[str]]) -> List[str]:
    if not raw_sites:
        return []

    sites: List[str] = []
    for entry in raw_sites:
        if "," in entry:
            sites.extend(part.strip() for part in entry.split(",") if part.strip())
        else:
            sites.append(entry.strip())

    normalized = []
    for h in sites:
        h = h.lower().strip()
        if h.startswith(("http://", "https://")):
            h = h.split("://", 1)[-1]
        h = h.removeprefix("www.").rstrip("/")
        if h and "." in h:
            normalized.append(h)

    return normalized


# ============================================================
# Main Pipeline
# ============================================================


def get_args() -> argparse.Namespace:
    DEFAULT_QUERY = "Latest top anime releases 2026"
    parser = argparse.ArgumentParser(
        description="Semantic Search + Adaptive Crawl (Embedding Strategy)"
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help=f"Search query (default: '{DEFAULT_QUERY}')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=8,
        help="Number of top semantic matches to select (default: 8)",
    )
    parser.add_argument(
        "--max-search-results",
        "-m",
        type=int,
        default=10,
        help="Maximum number of raw results to fetch from SearXNG (default: 10)",
    )

    parser.add_argument(
        "-s",
        "--site",
        action="append",
        dest="sites",
        type=str,
        default=None,
        help=(
            "Restrict results to one or more domains. "
            "Use multiple times or comma-separated. "
            "Example: -s github.com -s docs.python.org "
            "or -s github.com,docs.python.org"
        ),
    )

    return parser.parse_args()


def print_startup_info(args: argparse.Namespace, effective_query: str, embed_url: str):
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan", width=18)
    table.add_column()

    table.add_row("Query", f"[i]{args.query}[/]")
    table.add_row("Effective query", f"[i]{effective_query}[/]")
    table.add_row("Top K (seeds)", f"[green]{args.top_k}[/]")
    table.add_row("Max search results", f"[magenta]{args.max_search_results}[/]")
    table.add_row("Embedding", embed_url)
    if args.sites:
        table.add_row("Sites filter", ", ".join(args.sites))

    console.print(
        Panel(
            table,
            title="[bold]Semantic + Adaptive Crawl",
            border_style="bright_blue",
            padding=(1, 2),
        )
    )


async def search_seed_results(
    query: str,
    searxng_base_url: str = os.getenv("SEARXNG_URL"),
    timeout: float = 12.0,
    max_results: int = 10,
) -> List[dict]:
    console.print(f"[bold cyan]SearXNG[/]  →  [i]{query}[/i]", style="dim")

    with console.status("[bold green]Querying SearXNG...", spinner="dots"):
        params = {
            "q": query,
            "format": "json",
            "pageno": 1,
            "language": "en",
            "categories": "general",
        }

        query_string = urlencode(params)
        full_url = (
            urljoin(searxng_base_url.rstrip("/") + "/", "search") + "?" + query_string
        )

        console.print("[dim bright_black]SearXNG full request URL:[/]", style="dim")
        console.print(f"[blue underline]{full_url}[/blue underline]", soft_wrap=True)
        console.print("")

        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            try:
                resp = await client.get(full_url)
                resp.raise_for_status()
                data = resp.json()

                results = []
                for r in data.get("results", []):
                    if r.get("url", "").startswith("http"):
                        results.append(
                            {
                                "url": r["url"],
                                "title": r.get("title", ""),
                                "snippet": r.get("content", ""),
                            }
                        )

                return results[:max_results]

            except Exception as e:
                console.print(f"[bold red]ERROR[/] SearXNG request failed: {e}")
                return []


async def semantic_search_results(
    query: str,
    top_k: int = 8,
    max_search_results: int = 10,
    sites: Optional[List[str]] = None,
    embed_url: Optional[str] = None,
) -> List[SemanticResult]:
    """Main semantic search pipeline using embeddings for reranking."""

    if embed_url is None:
        embed_url = os.getenv("LLAMA_CPP_EMBED_URL")

    if not embed_url:
        console.print("[bold red]Missing environment variable:[/] LLAMA_CPP_EMBED_URL")
        return []

    normalized_sites = normalize_sites(sites)

    effective_query = query
    if normalized_sites:
        site_clause = " OR ".join(f"site:{domain}" for domain in normalized_sites)
        effective_query = f"{query} {site_clause}"

    print_startup_info(
        argparse.Namespace(
            query=query, top_k=top_k, max_search_results=max_search_results, sites=sites
        ),
        effective_query,
        embed_url,
    )

    # Phase 1 — Seed Discovery
    console.rule("Phase 1 — Seed Discovery (SearXNG)", style="blue")
    raw_results = await search_seed_results(
        effective_query, max_results=max_search_results
    )

    if not raw_results:
        console.print("[yellow]No results found.[/yellow]")
        return []

    console.print(f"\n[b green]Fetched {len(raw_results)} search results[/b green]\n")

    # Phase 2 — Semantic Reranking
    console.rule("Phase 2 — Semantic Reranking", style="magenta")

    semantic_results = await semantic_seed_filter(query, raw_results, top_k=top_k)

    if not semantic_results:
        console.print("[yellow]No strong semantic matches found.[/]")
        return []

    console.print(
        f"\n[b green]Selected {len(semantic_results)} strongest semantic results[/b green]\n"
    )

    return semantic_results


# ============================================================
# Rich Logging for Final Results
# ============================================================


def print_final_results(results: List[SemanticResult], query: str):
    """Print final results with 4-column header + separate snippet box below each result. Title and URL should NOT wrap."""
    if not results:
        console.print("[yellow]No semantic results to display.[/yellow]")
        return

    console.rule(f"Final Semantic Results — Top {len(results)}", style="bright_green")

    for item in results:
        rank = item.get("rank", "")
        score = f"{item.get('score', 0):.4f}"
        title = (item.get("title", "") or "[dim]— no title —[/]").strip()
        url_raw = item.get("url", "") or ""
        url_display = f"[link={url_raw.strip()}]{url_raw.strip()}[/link]"

        # Limit snippet
        snippet = (item.get("snippet", "") or "").strip()
        if len(snippet) > 350:
            snippet = snippet[:347].rstrip() + "..."

        # 1. Compact 4-column table for metadata
        meta_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.HEAVY_HEAD,
            expand=True,
            padding=(0, 1),
        )
        meta_table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
        meta_table.add_column(
            "Score", justify="right", style="cyan", width=9, no_wrap=True
        )
        meta_table.add_column("Title", style="bold white", ratio=2, no_wrap=True)
        meta_table.add_column("URL", style="blue underline", ratio=3, no_wrap=True)

        meta_table.add_row(str(rank), score, title, url_display)

        # 2. Snippet as its own full-width Panel (true "box" below)
        snippet_panel = Panel(
            snippet if snippet else "[dim]— no snippet available —[/]",
            title="Snippet",
            title_align="left",
            border_style="green dim",
            padding=(1, 2),
            expand=True,
        )

        # Print them together
        console.print(meta_table)
        console.print(snippet_panel)
        console.print("")  # spacing between results

    console.print(
        Panel(
            f"[bold green]✓ Completed:[/] Found {len(results)} semantically relevant results "
            f'for query: [cyan]"{query}"[/cyan]',
            border_style="bright_green",
            padding=(1, 2),
        )
    )


# ============================================================
# Entry Point
# ============================================================


if __name__ == "__main__":
    args = get_args()

    console.print("\n[bold bright_blue]Starting Semantic Search Pipeline[/]\n")

    results: List[SemanticResult] = asyncio.run(
        semantic_search_results(
            query=args.query,
            top_k=args.top_k,
            max_search_results=args.max_search_results,
            sites=args.sites,
        )
    )

    # Rich formatted output
    print_final_results(results, args.query)
