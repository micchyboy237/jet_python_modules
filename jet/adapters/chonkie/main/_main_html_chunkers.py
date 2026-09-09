import argparse
import shutil
from pathlib import Path
from typing import List

from jet.adapters.chonkie.html_chunkers import chunk
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_args() -> argparse.Namespace:
    """Parse CLI arguments supporting multiple positional sources."""
    parser = argparse.ArgumentParser(
        description="HTML-aware chunker CLI. Accepts URLs, local file paths, and/or raw HTML strings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single URL
  python -m jet.adapters.chonkie.html_chunkers https://deepeval.com/docs/metrics-introduction

  # Multiple mixed sources
  python -m jet.adapters.chonkie.html_chunkers https://deepeval.com/docs/metrics-introduction https://deepeval.com/docs/metrics-faithfulness ./docs/guide.html "<h1>Inline</h1><p>Raw HTML</p>"

  # With overrides
  python -m jet.adapters.chonkie.html_chunkers ./docs/*.html --chunk-size 384 --semantic-model minishlab/potion-base-32M
""",
    )

    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more URLs, local HTML file paths, or raw HTML strings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=50,
        help="Min characters per prose chunk (default: 50).",
    )
    parser.add_argument(
        "--semantic-model",
        type=str,
        default="minishlab/potion-base-32M",
        help="Embedding model for semantic fallback (default: minishlab/potion-base-32M).",
    )
    parser.add_argument(
        "--table-rows", type=int, default=5, help="Rows per table chunk (default: 5)."
    )
    parser.add_argument(
        "--min-section-tokens",
        type=int,
        default=50,
        help="Min tokens before merging short sections (default: 50).",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window during URL scraping.",
    )
    parser.add_argument(
        "--use-cache", action="store_true", help="Enable Playwright response caching."
    )
    parser.add_argument(
        "--scroll-strategy",
        type=str,
        default="until_stable",
        choices=["until_stable", "bottom", "none"],
        help="Scroll strategy for dynamic pages (default: until_stable).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, run chunk(), write results to OUTPUT_DIR."""
    args = get_args()
    console.log(f"[cyan]📥 Sources ({len(args.sources)}):[/]")
    for s in args.sources:
        display = s if len(s) <= 80 else s[:77] + "..."
        console.log(f"   • {display}")
    console.log(
        f"[cyan]⚙️  Config:[/] chunk_size={args.chunk_size}, "
        f"min_chars={args.min_chars}, model={args.semantic_model}"
    )
    results = chunk(
        source=args.sources,
        chunk_size=args.chunk_size,
        min_chars_per_chunk=args.min_chars,
        semantic_model=args.semantic_model,
        table_rows_per_chunk=args.table_rows,
        headless=not args.no_headless,
        use_cache=args.use_cache,
        scroll_strategy=args.scroll_strategy,
        min_section_tokens=args.min_section_tokens,
    )
    if not results:
        console.log(
            "[yellow]⚠ No chunks produced. Check source validity and logs above.[/]"
        )
        return

    console.log(f"[bold green]✏️  Writing {len(results)} chunks to {OUTPUT_DIR}/[/]")

    table = Table(
        title="Saved Chunks",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        pad_edge=False,
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Category", style="green", min_width=12)
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Preview", max_width=40, overflow="ellipsis")
    table.add_column("Actions", width=10, justify="center")

    for idx, r in enumerate(results, start=1):
        filename = f"{idx:04d}_{r.element_category.lower()}.txt"
        out_path = OUTPUT_DIR / filename
        lines: List[str] = [
            "---",
            f"index: {idx}",
            f"element_category: {r.element_category}",
            f'breadcrumb: "{r.breadcrumb}"',
            f'source_url: "{r.source_url or ""}"',
            f"page_number: {r.page_number}",
            f"token_count: {r.chunk.token_count}",
            f"start_index: {r.chunk.start_index}",
            f"end_index: {r.chunk.end_index}",
            "---",
            "",
            r.chunk.text,
        ]
        out_path.write_text("\n".join(lines), encoding="utf-8")

        # Build preview (first 40 chars, single-line, escape Rich markup)
        raw_preview = (r.chunk.text or "").strip().replace("\n", " ")
        preview = raw_preview[:40] + ("…" if len(raw_preview) > 40 else "")
        preview = preview.replace("[", "\\[").replace("]", "\\]")

        # File open link (file:// URI)
        file_uri = out_path.resolve().as_uri()
        file_link = f"[link={file_uri}]📄[/link]"

        # Source URL link (only if available)
        if r.source_url:
            safe_url = r.source_url.replace("[", "\\[").replace("]", "\\]")
            source_link = f"[link={safe_url}]🔗[/link]"
        else:
            source_link = "[dim]—[/dim]"

        table.add_row(
            str(idx),
            r.element_category,
            str(r.chunk.token_count),
            preview,
            f"{file_link}  {source_link}",
        )

    console.print(table)
    console.log(
        f"\n[bold green]✔ Done:[/] {len(results)} chunks written to [cyan]{OUTPUT_DIR}[/]"
    )


if __name__ == "__main__":
    main()
