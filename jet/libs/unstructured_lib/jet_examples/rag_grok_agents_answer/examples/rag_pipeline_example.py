import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_pipeline import (
    RAGPipeline,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create one rich console instance to use everywhere in this script
console = Console()

pipeline = RAGPipeline()

query = "What are the key features in the docs?"

console.rule("Starting ingestion", style="bold blue")
console.print("[cyan]Ingesting documents from path...[/cyan]")
pipeline.ingest(
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/crawl4ai/docs/md_v2/advanced"
)  # or list of files

console.rule("Query", style="bold magenta")
console.print(f"[yellow]Running query:[/yellow] {query}")
answer = pipeline.query(query)

console.rule("Result", style="bold green")
console.print("[bold green]Answer:[/bold green]")
console.print(
    answer, markup=False
)  # markup=False prevents interpreting answer as rich markup
# or: console.print(Panel(answer, title="Answer", border_style="green", padding=(1,2)))

save_file({"query": query}, OUTPUT_DIR / "input.json")
save_file(answer, OUTPUT_DIR / "answer.md")
