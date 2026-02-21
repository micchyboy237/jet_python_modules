import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_vectorstore import (
    ChromaVectorStore,
)
from rich.console import Console

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

console = Console()
vector_store = ChromaVectorStore(collection_name="example_collection")

console.rule("VectorStore Example", style="bold blue")

documents = [
    {"text": "RAG pipelines combine retrieval and generation.", "metadata": {"id": 1}},
    {"text": "Embeddings represent text as dense vectors.", "metadata": {"id": 2}},
]

# Fake embeddings for demonstration (must match dimension of real model in real usage)
embeddings = [[0.1] * 10, [0.2] * 10]

console.print("[cyan]Adding documents to vector store...[/cyan]")
vector_store.add_documents(documents, embeddings)

console.print("[cyan]Running similarity search...[/cyan]")
query_embedding = [0.1] * 10
results = vector_store.similarity_search(query_embedding, k=2)

console.print(f"[green]Retrieved {len(results)} results[/green]")

save_file(
    {
        "stored_documents": documents,
        "query_embedding": query_embedding,
        "results": results,
    },
    OUTPUT_DIR / "vectorstore_results.json",
)
