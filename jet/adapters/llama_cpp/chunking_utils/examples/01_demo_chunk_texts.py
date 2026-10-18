"""
Demo: Basic Text Chunking

Shows how to split text into token-sized chunks using chunk_texts.
"""

import shutil
from pathlib import Path

from jet.adapters.llama_cpp.chunking_utils import chunk_texts
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DOC_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/scrapers/playwright/generated/run_playwright_snapshot/snapshot.md"


def main():
    console.print(Panel.fit("🚀 Starting Basic Chunking Demo", style="bold cyan"))

    with open(DOC_PATH, "r", encoding="utf-8") as f:
        sample_text = f.read()

    console.print(
        f"[green]✓[/green] Loaded sample text ({len(sample_text)} characters)"
    )

    # Chunk with strict sentence boundaries and small size for demonstration
    console.print("\n[yellow]⚙️  Chunking parameters:[/yellow]")
    console.print("  • Chunk size: 64 tokens")
    console.print("  • Overlap: 8 tokens")
    console.print("  • Strict sentences: True")
    console.print("  • Min chunk size: 10 tokens")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking text...", total=None)

        chunks = chunk_texts(
            texts=sample_text,
            chunk_size=64,  # Max tokens per chunk
            chunk_overlap=8,  # Overlap tokens between chunks
            strict_sentences=True,
            min_chunk_size=10,  # Merge tiny trailing chunks
            show_progress=False,  # We're using our own progress indicator
        )

        progress.update(task, completed=True)

    console.print(f"\n[bold green]✓ Generated {len(chunks)} chunks[/bold green]\n")

    # Save each chunk to a file and display summary
    saved_files = []

    for i, chunk in enumerate(chunks):
        # Create filename with zero-padded index
        chunk_filename = f"chunk_{i + 1:03d}.txt"
        chunk_path = OUTPUT_DIR / chunk_filename

        # Save chunk to file
        with open(chunk_path, "w", encoding="utf-8") as f:
            f.write(chunk)

        saved_files.append(chunk_path)

        # Display chunk preview
        preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
        panel = Panel(
            Text(preview, style="white"),
            title=f"[bold blue]Chunk {i + 1}[/bold blue]",
            subtitle=f"[dim]{len(chunk)} chars | {chunk_filename}[/dim]",
            border_style="blue",
        )
        console.print(panel)

    # Display summary table of saved files
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]📁 Saved Files Summary[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("File Name", style="green")
    table.add_column("Size", style="yellow", justify="right")
    table.add_column("Resource Link", style="blue")

    for i, file_path in enumerate(saved_files, 1):
        file_size = file_path.stat().st_size
        base_name = file_path.name
        resource_link = f"file://{file_path.resolve()}"

        table.add_row(
            str(i),
            base_name,
            f"{file_size:,} bytes",
            resource_link,
        )

    console.print(table)

    console.print(f"\n[dim]All files saved to: {OUTPUT_DIR.resolve()}[/dim]")


if __name__ == "__main__":
    main()
