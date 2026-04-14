import argparse
import asyncio
import os
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlencode, urljoin

import httpx
import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


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


def make_scoring_table(scored: List[tuple[str, float]], top_k: int) -> Table:
    table = Table(
        title="Top Semantic Seeds (Sorted by Similarity)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("URL", style="green")

    for i, (url, score) in enumerate(scored[:top_k], 1):
        table.add_row(f"{i}", f"{score:.3f}", url)
    return table


async def semantic_seed_filter(
    query: str,
    results: List[dict],
    top_k: int = 8,
    embed_url: str = os.getenv("LLAMA_CPP_EMBED_URL"),
) -> List[str]:
    if not results:
        return []

    console.rule("Semantic Reranking", style="bright_blue")

    task_description = "[cyan]Computing embeddings..."
    with Live(console=console, refresh_per_second=8) as live:
        live.update(Panel(task_description, style="bold cyan"))

        texts = [query] + [f"{r['title']} {r['snippet']}" for r in results]
        embeddings = await embed_texts(texts, base_url=embed_url)

        live.update(
            Panel(
                "[green]Embeddings ready — calculating similarities...",
                style="bold green",
            )
        )

        # Calculate similarity scores
        scored = []
        for r, emb in zip(results, embeddings[1:]):
            score = cosine_similarity(embeddings[0], emb)
            scored.append((r["url"], score))

        # === SORT BY SIMILARITY SCORE IN DESCENDING ORDER ===
        scored.sort(key=lambda x: x[1], reverse=True)

        live.update(
            Group(
                Panel(
                    "[green]Similarity scores calculated and sorted", style="bold green"
                ),
                make_scoring_table(scored, top_k),
            )
        )

    # Small delay so user can see the table
    await asyncio.sleep(0.6)

    # Return top_k URLs (already sorted descending by score)
    return [url for url, _ in scored[:top_k]]


# ============================================================
# Site Normalization
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

        console.print(
            "[dim bright_black]SearXNG full request URL:[/]",
            style="dim",
        )
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
) -> List[str]:
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
            query=query,
            top_k=top_k,
            max_search_results=max_search_results,
            sites=sites,
        ),
        effective_query,
        embed_url,
    )

    # ── Phase 1 ─────────────────────────────────────────────
    console.rule("Phase 1 — Seed Discovery (SearXNG)", style="blue")
    raw_results = await search_seed_results(
        effective_query,
        max_results=max_search_results,
    )

    if not raw_results:
        console.print("[yellow]No results found.[/yellow]")
        return []

    console.print(f"\n[b green]Fetched {len(raw_results)} search results[/b green]\n")

    # ── Phase 2 ─────────────────────────────────────────────
    console.rule("Phase 2 — Semantic Reranking", style="magenta")

    seed_urls = await semantic_seed_filter(
        query,
        raw_results,
        top_k=top_k,
    )

    if not seed_urls:
        console.print("[yellow]No strong semantic matches found.[/]")
        return []

    console.print(
        f"\n[b green]Selected {len(seed_urls)} strongest seed URLs[/b green]\n"
    )

    return seed_urls


if __name__ == "__main__":
    args = get_args()
    asyncio.run(
        semantic_search_results(
            query=args.query,
            top_k=args.top_k,
            max_search_results=args.max_search_results,
            sites=args.sites,
        )
    )
