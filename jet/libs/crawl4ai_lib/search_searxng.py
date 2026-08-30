"""Semantic search pipeline for crawl4ai_lib.

Phase 1: Searches via SearXNG using jet.search.searxng.async_search_searxng
         (gains Redis caching, deduplication, filtering, score sorting).
Phase 2: Reranks results using embeddings from jet.adapters.llama_cpp
         (shared factory client, scoring utils, centralized config).
"""

import argparse
import asyncio
from dataclasses import dataclass
from typing import List, Optional, TypedDict

import numpy as np
from jet.adapters.llama_cpp.factory import get_async_embedding_client
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.libs.crawl4ai_lib.config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_SEARCH_RESULTS,
    DEFAULT_TOP_K,
    EMBED_BASE_URL,
    EMBED_MODEL,
    EMBED_REQUEST_TIMEOUT,
    SEARXNG_REQUEST_TIMEOUT,
    SEARXNG_URL,
)
from jet.logger import logger
from jet.search.searxng import async_search_searxng
from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

console = Console()


class SemanticResult(TypedDict):
    """Typed dictionary for semantic search/reranking results."""

    rank: int
    score: float
    title: str
    url: str
    snippet: str


@dataclass
class AppConfig:
    query: str
    top_k: int
    sites: Optional[List[str]]


async def embed_texts(
    texts: List[str],
    base_url: str = EMBED_BASE_URL,
    embed_model: str = EMBED_MODEL,
    timeout: float = EMBED_REQUEST_TIMEOUT,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[np.ndarray]:
    """Embed texts using the shared async OpenAI client from factory.

    Includes connectivity pre-check and strict timeout to prevent silent hangs.
    Returns embeddings in the same order as input texts.
    """
    if not texts:
        return []

    logger.info(
        f"embed_texts: n_texts={len(texts)}, model={embed_model}, base_url={base_url}"
    )

    console.print(f"[dim]Verifying embedding server at {base_url}...[/dim]")
    try:
        check_client = get_async_embedding_client(
            base_url=base_url, timeout=5.0, max_retries=0
        )
        await check_client.models.list()
        console.print("[green]✓ Embedding server reachable[/green]")
    except Exception as e:
        logger.error(f"embed_texts: server unreachable at {base_url} - {e}")
        console.print(f"[bold red]✗ Embedding server unreachable: {e}[/bold red]")
        raise ConnectionError(
            f"Embedding server at {base_url} is not responding. "
            f"Check network/server status."
        ) from e

    client = get_async_embedding_client(
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
    )
    try:
        logger.info(
            f"embed_texts: sending request (timeout={timeout}s, retries={max_retries})"
        )
        response = await client.embeddings.create(
            input=texts,
            model=embed_model,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [
            np.array(item.embedding, dtype=np.float32) for item in sorted_data
        ]
        logger.info(f"embed_texts: successfully embedded {len(embeddings)} texts")
        return embeddings
    except Exception as e:
        logger.error(f"embed_texts: failed after retries - {type(e).__name__}: {e}")
        console.print(f"[bold red]✗ Embedding failed: {e}[/bold red]")
        raise


async def semantic_seed_filter(
    query: str,
    results: List[dict],
    top_k: int = DEFAULT_TOP_K,
    embed_url: str = EMBED_BASE_URL,
    embed_model: str = EMBED_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[SemanticResult]:
    """Perform semantic reranking using shared embedding and scoring utils.

    Falls back to unranked results if embedding fails, so the crawl can continue.
    """
    if not results:
        logger.warning("semantic_seed_filter: empty results list, returning []")
        return []

    logger.info(
        f"semantic_seed_filter: query='{query[:80]}...', "
        f"n_results={len(results)}, top_k={top_k}"
    )
    console.rule("Semantic Reranking", style="bright_blue")

    try:
        task_description = "[cyan]Computing embeddings..."
        with Live(console=console, refresh_per_second=8) as live:
            live.update(Panel(task_description, style="bold cyan"))

            texts = [query] + [
                f"{r.get('title', '')} {r.get('snippet', '')}" for r in results
            ]
            embeddings = await embed_texts(
                texts,
                base_url=embed_url,
                embed_model=embed_model,
                max_retries=max_retries,
            )

            live.update(
                Panel(
                    "[green]Embeddings ready — calculating similarities...",
                    style="bold green",
                )
            )

            scored: List[SemanticResult] = []
            query_emb = embeddings[0]

            for i, (r, emb) in enumerate(zip(results, embeddings[1:]), 1):
                score = cosine_similarity(query_emb, emb)
                scored.append(
                    {
                        "rank": i,
                        "score": score,
                        "title": r.get("title", "").strip(),
                        "url": r.get("url", "").strip(),
                        "snippet": r.get("snippet", "").strip(),
                    }
                )

            scored.sort(key=lambda x: x["score"], reverse=True)
            for i, item in enumerate(scored, 1):
                item["rank"] = i

            live.update(
                Panel(
                    "[green]Similarity scores calculated and sorted",
                    style="bold green",
                )
            )
            await asyncio.sleep(0.6)

        logger.info(
            f"semantic_seed_filter: selected {min(top_k, len(scored))}/{len(scored)} "
            f"results, top_score={scored[0]['score']:.4f}"
            if scored
            else "semantic_seed_filter: no scored results"
        )
        return scored[:top_k]

    except (ConnectionError, Exception) as e:
        logger.warning(
            f"semantic_seed_filter: embedding failed, returning raw results "
            f"unranked - {e}"
        )
        console.print(
            "[yellow]⚠ Semantic reranking unavailable, using raw SearXNG order[/yellow]"
        )
        return [
            {
                "rank": i + 1,
                "score": 0.0,
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("snippet", ""),
            }
            for i, r in enumerate(results[:top_k])
        ]


def normalize_sites(raw_sites: Optional[List[str]]) -> List[str]:
    """Normalize site filter strings into clean domain names."""
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
        default=DEFAULT_TOP_K,
        help=f"Number of top semantic matches to select (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--max-search-results",
        "-m",
        type=int,
        default=DEFAULT_MAX_SEARCH_RESULTS,
        help=(
            f"Maximum raw results from SearXNG (default: {DEFAULT_MAX_SEARCH_RESULTS})"
        ),
    )
    parser.add_argument(
        "--max-retries",
        "-r",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries for HTTP requests (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "-s",
        "--site",
        action="append",
        dest="sites",
        type=str,
        default=None,
        help=(
            "Restrict results to domains. Use multiple times or comma-separated. "
            "Example: -s github.com -s docs.python.org"
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
    table.add_row("Max retries", f"[magenta]{args.max_retries}[/]")
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
    searxng_base_url: str = SEARXNG_URL,
    timeout: float = SEARXNG_REQUEST_TIMEOUT,
    max_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[dict]:
    """Fetch raw search results from SearXNG using shared async_search_searxng.

    Gains Redis caching, deduplication, relevance filtering, and score sorting
    from jet.search.searxng without duplicating any logic.
    """
    logger.info(
        f"search_seed_results: query='{query[:80]}...', "
        f"url={searxng_base_url}, max_results={max_results}"
    )
    console.print(f"[bold cyan]SearXNG[/] → [i]{query}[/i]", style="dim")

    with console.status("[bold green]Querying SearXNG...", spinner="dots"):
        try:
            search_results = await async_search_searxng(
                query=query,
                query_url=searxng_base_url,
                count=max_results,
                max_retries=max_retries,
                timeout=timeout,
                use_cache=True,
            )
            # Normalize SearchResult fields to simple dicts for downstream use
            results = [
                {
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "snippet": r.get("content", ""),
                }
                for r in search_results
                if r.get("url", "").startswith("http")
            ]
            logger.info(
                f"search_seed_results: fetched {len(results)} results "
                f"(requested max={max_results})"
            )
            return results
        except Exception as e:
            logger.error(f"search_seed_results: SearXNG request failed - {e}")
            console.print(f"[bold red]ERROR[/] SearXNG request failed: {e}")
            return []


async def semantic_search_results(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    sites: Optional[List[str]] = None,
    embed_url: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> List[SemanticResult]:
    """Main semantic search pipeline using embeddings for reranking."""
    if embed_url is None:
        embed_url = EMBED_BASE_URL
    if not embed_url:
        logger.error("semantic_search_results: missing EMBED_BASE_URL config")
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
            max_retries=max_retries,
        ),
        effective_query,
        embed_url,
    )

    console.rule("Phase 1 — Seed Discovery (SearXNG)", style="blue")
    raw_results = await search_seed_results(
        effective_query,
        max_results=max_search_results,
        max_retries=max_retries,
    )
    if not raw_results:
        logger.warning("semantic_search_results: no raw results from SearXNG")
        console.print("[yellow]No results found.[/yellow]")
        return []

    console.print(f"\n[b green]Fetched {len(raw_results)} search results[/b green]\n")

    console.rule("Phase 2 — Semantic Reranking", style="magenta")
    semantic_results = await semantic_seed_filter(
        query,
        raw_results,
        top_k=top_k,
        embed_url=embed_url,
        max_retries=max_retries,
    )
    if not semantic_results:
        logger.warning("semantic_search_results: no strong semantic matches")
        console.print("[yellow]No strong semantic matches found.[/]")
        return []

    logger.info(
        f"semantic_search_results: completed with {len(semantic_results)} results"
    )
    console.print(
        f"\n[b green]Selected {len(semantic_results)} strongest semantic "
        f"results[/b green]\n"
    )
    return semantic_results


def print_final_results(results: List[SemanticResult], query: str):
    """Print final results with 4-column header + separate snippet box."""
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
        snippet = (item.get("snippet", "") or "").strip()
        if len(snippet) > 350:
            snippet = snippet[:347].rstrip() + "..."

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

        snippet_panel = Panel(
            snippet if snippet else "[dim]— no snippet available —[/]",
            title="Snippet",
            title_align="left",
            border_style="green dim",
            padding=(1, 2),
            expand=True,
        )
        console.print(meta_table)
        console.print(snippet_panel)
        console.print("")

    console.print(
        Panel(
            f"[bold green]✓ Completed:[/] Found {len(results)} semantically "
            f'relevant results for query: [cyan]"{query}"[/cyan]',
            border_style="bright_green",
            padding=(1, 2),
        )
    )


if __name__ == "__main__":
    args = get_args()
    console.print("\n[bold bright_blue]Starting Semantic Search Pipeline[/]\n")
    results: List[SemanticResult] = asyncio.run(
        semantic_search_results(
            query=args.query,
            top_k=args.top_k,
            max_search_results=args.max_search_results,
            sites=args.sites,
            max_retries=args.max_retries,
        )
    )
    print_final_results(results, args.query)
