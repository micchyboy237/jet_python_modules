from jet.adapters.llama_cpp.hybrid_search import HybridSearch
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from rich.console import Console
from rich.table import Table

if __name__ == "__main__":
    model: LLAMACPP_EMBED_KEYS = "nomic-embed-text"

    documents = [
        "Fresh organic apples from local farms",
        "Handpicked strawberries sweet and juicy",
        "Premium quality oranges rich in vitamin C",
        "Crisp lettuce perfect for salads",
        "Organic bananas ripe and ready to eat",
    ]

    queries = ["organic fruit", "sweet strawberries", "fresh salad ingredients"]

    hybrid = HybridSearch.from_documents(
        documents=documents,
        model=model,
    )

    for num, query in enumerate(queries, start=1):
        results = hybrid.search(query, top_k=5)

        console = Console()
        table = Table(title=f"Hybrid Results for: {query!r}")
        table.add_column("Rank", justify="right", style="cyan")
        table.add_column("Hybrid", justify="right")
        table.add_column("Dense", justify="right")
        table.add_column("Sparse", justify="right")
        table.add_column("Category", style="bold")
        table.add_column("Level", justify="right", style="dim cyan")
        table.add_column("ID")
        table.add_column("Preview", style="dim")

        for res in results:
            preview = res["text"][:80] + "..." if len(res["text"]) > 80 else res["text"]
            table.add_row(
                str(res["rank"]),
                f"{res['hybrid_score']:.4f}",
                f"{res['dense_score']:.3f}",
                f"{res['sparse_score']:.3f}",
                res["category"],
                str(res["category_level"]),
                res["id"] or "-",
                preview,
            )

        console.print(table)
