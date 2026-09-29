import json
import shutil
from pathlib import Path

from jet.adapters.llama_cpp.config import LLM_MODEL
from jet.adapters.unstructured.document_parser import parse_document
from rich.console import Console

console = Console()

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


max_tokens = 450
overlap_tokens = 50
model = LLM_MODEL

pdf_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Apps/my-jobs/data/Resume Latest - Jethro Estrada.pdf"
query = "What is your educational background?"

console.print("[bold blue]Parsing document...[/bold blue]")
elements = parse_document(
    str(pdf_path),
    chunk_max_tokens=max_tokens,
    chunk_overlap_tokens=overlap_tokens,
    model=model,
)

console.print(f"[green]✓ Parsed {len(elements)} elements[/green]")

elements_path = OUTPUT_DIR / f"elements.json"
with open(elements_path, "w", encoding="utf-8") as f:
    json.dump(elements, f, indent=2, ensure_ascii=False, default=str)

console.print(
    f"[bold green]✓ Saved elements to [/bold green]"
    f"[link=file://{elements_path}]{elements_path.name}[/link]"
)
