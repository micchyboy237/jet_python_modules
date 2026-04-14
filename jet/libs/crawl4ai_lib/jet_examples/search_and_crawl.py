import asyncio
import os
import time

from crawl4ai import AdaptiveCrawler, AsyncWebCrawler
from jet.libs.crawl4ai_lib.adaptive_config import get_adaptive_config
from jet.libs.crawl4ai_lib.search_searxng import (
    get_args,
    normalize_sites,
    print_startup_info,
    search_seed_results,
    semantic_seed_filter,
)
from rich.console import Console
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


# ============================================================
# Main Pipeline
# ============================================================


async def main():
    args = get_args()

    embed_url = os.getenv("LLAMA_CPP_EMBED_URL")

    if not embed_url:
        console.print("[bold red]Missing environment variable:[/] LLAMA_CPP_EMBED_URL")
        return

    normalized_sites = normalize_sites(args.sites)

    effective_query = args.query
    if normalized_sites:
        site_clause = " OR ".join(f"site:{domain}" for domain in normalized_sites)
        effective_query = f"{args.query} {site_clause}"

    print_startup_info(args, effective_query, embed_url)

    # ── Phase 1 ─────────────────────────────────────────────
    console.rule("Phase 1 — Seed Discovery (SearXNG)", style="blue")
    raw_results = await search_seed_results(
        effective_query,
        max_results=args.max_search_results,
    )

    if not raw_results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[b green]Fetched {len(raw_results)} search results[/b green]\n")

    # ── Phase 2 ─────────────────────────────────────────────
    console.rule("Phase 2 — Semantic Reranking", style="magenta")

    seed_urls = await semantic_seed_filter(
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
