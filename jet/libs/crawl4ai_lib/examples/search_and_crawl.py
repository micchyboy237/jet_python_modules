import argparse
import asyncio
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlencode, urljoin

import httpx
import numpy as np
from crawl4ai import AdaptiveCrawler, AsyncWebCrawler
from jet.libs.crawl4ai_lib.adaptive_config import get_adaptive_config
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# ============================================================
# Utility: Cosine Similarity
# ============================================================

console = Console()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ============================================================
# SearXNG Fetch
# ============================================================


async def fetch_seed_results_from_searxng(
    searxng_base_url: str,
    query: str,
    timeout: float = 12.0,
) -> List[dict]:
    """
    Fetch full search results (url + title + snippet)
    """
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
            "[dim bright_black]SearXNG full request URL:[/dim bright_black]",
            style="dim",
        )
        console.print(f"[blue underline]{full_url}[/blue underline]", soft_wrap=True)
        console.print("")  # empty line for visual separation

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
                return results

            except Exception as e:
                console.print(f"[bold red]ERROR[/] SearXNG request failed: {e}")
                return []


# ============================================================
# Embedding API Call (OpenAI-compatible llama.cpp)
# ============================================================


async def embed_texts(
    base_url: str,
    texts: List[str],
) -> List[np.ndarray]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{base_url}/embeddings",
            json={
                "model": "nomic-embed-text-v2-moe",
                "input": texts,
            },
        )
        response.raise_for_status()

        data = response.json()

        return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]


# ============================================================
# Semantic Seed Reranking
# ============================================================


def make_scoring_table(scored: List[Tuple[str, float]], top_k: int) -> Table:
    table = Table(
        title="Top Semantic Seeds", show_header=True, header_style="bold magenta"
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("URL", style="green")

    for i, (url, score) in enumerate(scored[:top_k], 1):
        table.add_row(f"{i}", f"{score:.3f}", url)
    return table


async def semantic_seed_filter(
    embed_url: str,
    query: str,
    results: List[dict],
    top_k: int,
) -> List[str]:
    if not results:
        return []

    console.rule("Semantic Reranking", style="bright_blue")

    task_description = "[cyan]Computing embeddings..."
    with Live(console=console, refresh_per_second=8) as live:
        live.update(Panel(task_description, style="bold cyan"))

        texts = [query] + [f"{r['title']} {r['snippet']}" for r in results]
        embeddings = await embed_texts(embed_url, texts)

        live.update(
            Panel(
                "[green]Embeddings ready — calculating similarities...",
                style="bold green",
            )
        )

        scored = []
        for r, emb in zip(results, embeddings[1:]):
            score = cosine_similarity(embeddings[0], emb)
            scored.append((r["url"], score))

        live.update(
            Group(
                Panel("[green]Similarity scores calculated", style="bold green"),
                make_scoring_table(scored, top_k),
            )
        )

    # Small delay so user can see the table
    await asyncio.sleep(0.6)

    scored.sort(key=lambda x: x[1], reverse=True)

    return [url for url, _ in scored[:top_k]]


# ============================================================
# CLI + AppConfig (dataclass)
# ============================================================


@dataclass
class AppConfig:
    query: str
    top_seeds: int
    top_k: int
    sites: Optional[List[str]]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic Search + Adaptive Crawl (Embedding Strategy)"
    )

    parser.add_argument("query")
    parser.add_argument("--top-seeds", "-n", type=int, default=8)
    parser.add_argument("--top-k", "-k", type=int, default=5)

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


def print_startup_info(args: argparse.Namespace, effective_query: str, embed_url: str):
    table = Table.grid(expand=True)
    table.add_column(style="bold cyan", width=18)
    table.add_column()

    table.add_row("Query", f"[i]{args.query}[/]")
    table.add_row("Effective query", f"[i]{effective_query}[/]")
    table.add_row("Top seeds", f"[green]{args.top_seeds}[/]")
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


async def main():
    args = get_args()

    searxng_url = os.getenv("SEARXNG_URL")
    embed_url = os.getenv("LLAMA_CPP_EMBED_URL")

    if not searxng_url or not embed_url:
        console.print(
            "[bold red]Missing environment variables:[/] SEARXNG_URL and/or LLAMA_CPP_EMBED_URL"
        )
        return

    normalized_sites = normalize_sites(args.sites)

    effective_query = args.query
    if normalized_sites:
        site_clause = " OR ".join(f"site:{domain}" for domain in normalized_sites)
        effective_query = f"{args.query} {site_clause}"

    print_startup_info(args, effective_query, embed_url)

    # ── Phase 1 ─────────────────────────────────────────────
    console.rule("Phase 1 — Seed Discovery (SearXNG)", style="blue")
    raw_results = await fetch_seed_results_from_searxng(
        searxng_url,
        effective_query,
    )

    if not raw_results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[b green]Fetched {len(raw_results)} search results[/b green]\n")

    # ── Phase 2 ─────────────────────────────────────────────
    console.rule("Phase 2 — Semantic Reranking", style="magenta")

    seed_urls = await semantic_seed_filter(
        embed_url,
        args.query,
        raw_results,
        args.top_seeds,
    )

    if not seed_urls:
        console.print("[yellow]No strong semantic matches found.[/]")
        return

    console.print(
        f"\n[b green]Selected {len(seed_urls)} strongest seed URLs[/b green]\n"
    )

    # ── Phase 3 ─────────────────────────────────────────────
    console.rule("Phase 3 — Adaptive Embedding Crawl", style="green")

    adaptive_config = get_adaptive_config(
        strategy="embedding",
        n_query_variations=8,
        max_pages=15,
        top_k_links=3,
        min_gain_threshold=0.05,
        embedding_k_exp=3.0,
        embedding_min_confidence_threshold=0.1,
        embedding_validation_min_score=0.4,
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        adaptive = AdaptiveCrawler(crawler, adaptive_config)

        with progress:
            main_task = progress.add_task(
                "[cyan]Adaptive crawling...", total=len(seed_urls)
            )

            for idx, url in enumerate(seed_urls, 1):
                task = progress.add_task(
                    f"[yellow]Digesting {url[:60]}{'...' if len(url) > 60 else ''}",
                    total=None,
                )
                start = time.monotonic()

                await adaptive.digest(start_url=url, query=args.query)

                duration = time.monotonic() - start
                progress.update(
                    task,
                    completed=1,
                    description=f"[green]Done[/] {url[:60]} ({duration:.1f}s)",
                )
                progress.advance(main_task)

            progress.update(main_task, completed=len(seed_urls))

    # ── Results ─────────────────────────────────────────────
    console.rule("Most Relevant Pages", style="bright_green")

    relevant_pages = adaptive.get_relevant_content(top_k=args.top_k)

    table = Table(show_header=True, header_style="bold green")
    table.add_column("#", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Title / URL")

    for i, page in enumerate(relevant_pages, 1):
        title = page.get("title", "").strip() or "(no title)"
        url_short = page["url"][:70] + "…" if len(page["url"]) > 70 else page["url"]
        table.add_row(
            f"{i}", f"{page['score']:.1%}", f"[i]{title}[/i]\n[dim]{url_short}[/dim]"
        )

    console.print(table)

    console.print(
        Panel(
            f"[bold]Final pipeline confidence:  [green]{adaptive.confidence:.1%}[/green]",
            style="bold green",
            expand=False,
        )
    )

    if len(relevant_pages) == 0:
        console.print("[yellow]No pages passed the confidence threshold.[/yellow]")
    elif len(relevant_pages) < args.top_k:
        console.print(
            f"[yellow]Only {len(relevant_pages)}/{args.top_k} pages met criteria.[/yellow]"
        )


if __name__ == "__main__":
    asyncio.run(main())
