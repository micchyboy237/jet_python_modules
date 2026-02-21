import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_embedder import (
    LlamaCppEmbedder,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
embedder = LlamaCppEmbedder()

texts = [
    "This is a sample document chunk.",
    "Another chunk for embedding demonstration.",
]

console.rule("Embedding Example", style="bold blue")
console.print("[cyan]Embedding documents...[/cyan]")

embeddings = embedder.embed_documents(texts)

console.print(f"[green]Generated {len(embeddings)} embeddings[/green]")

save_file(
    {
        "texts": texts,
        "embeddings": embeddings,
    },
    OUTPUT_DIR / "embeddings.json",
)
