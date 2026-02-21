import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_processor import (
    DocumentProcessor,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
processor = DocumentProcessor()

file_path = (Path(__file__).parent.parent / "README.md").resolve()

console.rule("Document Processing", style="bold blue")
console.print(f"[cyan]Processing file:[/cyan] {file_path}")

chunks = processor.process_file(file_path)

console.print(f"[green]Produced {len(chunks)} chunks[/green]")

save_file(chunks, OUTPUT_DIR / "chunks.json")
