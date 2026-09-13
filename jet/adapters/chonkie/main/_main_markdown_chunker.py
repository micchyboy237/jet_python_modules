import argparse
import json
import shutil
from pathlib import Path
from typing import List

from jet.adapters.chonkie.markdown_chunker import chunk_markdown
from rich.console import Console
from rich.table import Table

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_args() -> argparse.Namespace:
    """Parse CLI arguments for markdown chunking."""
    parser = argparse.ArgumentParser(
        description="Markdown-aware chunker CLI. Accepts .md file paths or raw markdown strings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Single file
  python -m jet.adapters.chonkie.main._main_markdown_chunker ./docs/guide.md

  # Multiple files with custom chunk size
  python -m jet.adapters.chonkie.main._main_markdown_chunker ./docs/*.md --chunk-size 384

  # Raw markdown string
  python -m jet.adapters.chonkie.main._main_markdown_chunker "# Title\nSome content here" --lang en
""",
    )
    parser.add_argument(
        "sources",
        nargs="+",
        help="One or more .md file paths or raw markdown strings.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Target tokens per chunk (default: 512).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for recipe selection (default: en).",
    )
    parser.add_argument(
        "--keep-temp-file",
        action="store_true",
        help="Preserve temporary .md files created during processing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: parse args, run chunk_markdown(), write results to OUTPUT_DIR."""
    args = get_args()

    console.log(f"[cyan]📥 Sources ({len(args.sources)}):[/]")
    for s in args.sources:
        display = s if len(s) <= 80 else s[:77] + "..."
        console.log(f"   • {display}")

    console.log(f"[cyan]⚙️  Config:[/] chunk_size={args.chunk_size}, lang={args.lang}")

    chunks_dir = OUTPUT_DIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_manifest: List[dict] = []
    global_idx = 0

    for source in args.sources:
        source_path = Path(source)
        is_file = source_path.exists() and source_path.is_file()

        if is_file:
            try:
                text = source_path.read_text(encoding="utf-8")
                source_label = str(source_path.resolve())
                console.log(f"[green]📄 Processing file: {source_label}[/]")
            except Exception as e:
                console.log(f"[red]✖ Failed to read '{source}': {e}[/]")
                continue
        else:
            text = source
            source_label = "<raw_input>"
            console.log(f"[dim]📝 Processing raw markdown ({len(text):,} chars)[/]")

        try:
            result = chunk_markdown(
                text=text,
                chunk_size=args.chunk_size,
                lang=args.lang,
                keep_temp_file=args.keep_temp_file,
            )
        except Exception as e:
            console.log(f"[red]✖ Chunking failed for '{source_label}': {e}[/]")
            continue

        if not result.chunks:
            console.log(f"[yellow]⚠ No chunks produced for '{source_label}'[/]")
            continue

        console.log(
            f"[bold green]✔ {source_label}:[/] {len(result.chunks)} chunks, "
            f"{len(result.tables)} tables, {len(result.code_blocks)} code blocks, "
            f"{len(result.images)} images"
        )

        for chunk in result.chunks:
            global_idx += 1
            filename = f"{global_idx:03d}_markdown.txt"
            out_path = chunks_dir / filename

            lines: List[str] = [
                "---",
                f"index: {global_idx}",
                f'source: "{source_label}"',
                f"token_count: {chunk.token_count}",
                f"start_index: {chunk.start_index}",
                f"end_index: {chunk.end_index}",
                f"tables_extracted: {len(result.tables)}",
                f"code_blocks_extracted: {len(result.code_blocks)}",
                f"images_extracted: {len(result.images)}",
                "---",
                "",
                chunk.text,
            ]
            out_path.write_text("\n".join(lines), encoding="utf-8")

            all_manifest.append(
                {
                    "index": global_idx,
                    "filename": filename,
                    "source": source_label,
                    "token_count": chunk.token_count,
                    "start_index": chunk.start_index,
                    "end_index": chunk.end_index,
                    "text": chunk.text,
                }
            )

    if not all_manifest:
        console.log(
            "[yellow]⚠ No chunks produced overall. Check sources and logs above.[/]"
        )
        return

    # Write manifest
    manifest_path = OUTPUT_DIR / "chunks.json"
    manifest_path.write_text(json.dumps(all_manifest, indent=2), encoding="utf-8")
    console.log(f"[dim]📋 Saved manifest: {manifest_path}[/]")

    # Display summary table
    table = Table(
        title="Saved Markdown Chunks",
        show_header=True,
        header_style="bold cyan",
        border_style="dim",
        pad_edge=False,
    )
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Tokens", justify="right", width=7)
    table.add_column("Source", max_width=40, overflow="ellipsis")
    table.add_column("Preview", max_width=60, overflow="ellipsis")
    table.add_column("File", width=6, justify="center")

    for entry in all_manifest:
        raw_preview = (entry["text"] or "").strip().replace("\n", " ")
        preview = raw_preview[:60] + ("…" if len(raw_preview) > 60 else "")
        preview = preview.replace("[", "\\[").replace("]", "\\]")

        file_uri = (chunks_dir / entry["filename"]).resolve().as_uri()
        file_link = f"[link={file_uri}]📄[/link]"

        source_display = entry["source"]
        if len(source_display) > 40:
            source_display = "…" + source_display[-39:]

        table.add_row(
            str(entry["index"]),
            str(entry["token_count"]),
            source_display,
            preview,
            file_link,
        )

    console.print(table)
    console.log(
        f"[bold green]✔ Total:[/] {len(all_manifest)} chunks saved to {chunks_dir}/"
    )


if __name__ == "__main__":
    main()
