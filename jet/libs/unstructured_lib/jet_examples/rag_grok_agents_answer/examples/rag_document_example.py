import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_document import (
    RAGDocument,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

console.rule("RAGDocument Example", style="bold blue")

doc = RAGDocument(
    page_content="Example content for testing reusable document structure.",
    metadata={"source": "manual_test", "category": "demo"},
)

console.print("[green]Created RAGDocument instance[/green]")
console.print(doc)

save_file(
    {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    },
    OUTPUT_DIR / "document.json",
)
