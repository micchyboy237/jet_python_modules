"""
More realistic example with longer texts and richer metadata
"""

from jet.search.hybrid.retrieval_pipeline import Document, RetrievalPipeline
from rich.console import Console

console = Console()


def main():
    documents = [
        Document(
            id="py310-release",
            text="Python 3.10 was released in October 2021 and introduced structural pattern matching, parenthesized context managers, and better error messages.",
            metadata={
                "product": "python",
                "version": "3.10",
                "release_date": "2021-10-04",
                "category": "release-notes",
                "tags": ["pattern-matching", "error-messages"],
            },
        ),
        Document(
            id="py311-release",
            text="Python 3.11 (October 2022) brought up to 60% faster CPython, exception groups, tomllib for TOML parsing, and improved typing with Self and LiteralString.",
            metadata={
                "product": "python",
                "version": "3.11",
                "release_date": "2022-10-24",
                "category": "release-notes",
                "tags": ["performance", "typing", "exceptions"],
            },
        ),
        Document(
            id="rust-edition-2021",
            text="Rust 2021 edition introduced const generics, improved closure capture rules, and IntoIterator for arrays among other changes.",
            metadata={
                "product": "rust",
                "edition": "2021",
                "category": "language",
                "tags": ["const-generics", "closures"],
            },
        ),
    ]

    pipeline = RetrievalPipeline()
    console.rule("Indexing documents")
    pipeline.add_documents(documents)
    console.print(f"[green]→ {len(documents)} documents loaded[/green]\n")

    queries = [
        "python pattern matching",
        "faster python release",
        "rust const generics",
    ]

    for query in queries:
        console.rule(f"Query: {query}")
        results = pipeline.retrieve(query, top_k=2)

        for rank, doc in enumerate(results, 1):
            console.print(f"[bold cyan]{rank}.[/bold cyan] [yellow]{doc.id}[/yellow]")
            console.print(f"   {doc.text[:180]}{'...' if len(doc.text) > 180 else ''}")
            console.print(f"   [dim]metadata:[/dim] {doc.metadata}\n")


if __name__ == "__main__":
    main()
