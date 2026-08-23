"""Evaluate relevance using instruction-aware embeddings via llama.cpp server.

Equivalent to jet/models/tasks/evaluate_relevance.py but uses the OpenAI-compatible
embedding endpoint instead of local MLX inference. Reuses embed_utils for batched
embedding generation and scoring_utils for similarity computation.
"""

from typing import Union

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.embed_utils import embed
from jet.adapters.llama_cpp.scoring_utils import cosine_similarity
from jet.logger import logger
from tqdm import tqdm


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Format a query with an instruction prefix for instruct-tuned embedding models.

    Matches the template from the MLX reference implementation exactly.
    """
    return f"Instruct: {task_description}\nQuery:{query}"


def evaluate_relevance(
    queries: Union[str, list[str]],
    documents: list[str],
    task_description: str,
    model_name: str | None = None,
    max_length: int | None = None,
    batch_size: int | None = None,
    show_progress: bool = False,
    doc_batch_size: int = 50,
) -> list[list[float]]:
    """Compute relevance scores between queries and documents using instruction-aware embeddings.

    This is the llama.cpp adapter equivalent of jet/models/tasks/evaluate_relevance.py.
    Instead of local MLX inference, it uses the configured embedding server which handles
    tokenization, padding, pooling, and normalization server-side.

    Args:
        queries: Single query string or list of query strings.
        documents: List of document strings to score against each query.
        task_description: Instruction describing the retrieval task
            (e.g., "Given a web search query, retrieve relevant passages").
        model_name: Embedding model key. Defaults to EMBED_MODEL from config.
        max_length: Ignored. Kept for API compatibility with MLX version.
            Server handles truncation via its own context window.
        batch_size: Passed to embed() as the per-request batch size.
            Defaults to embed()'s internal default (16).
        show_progress: Whether to display progress bars.
        doc_batch_size: Ignored. Kept for API compatibility.
            embed() handles its own batching and deduplication.

    Returns:
        List of score lists. scores[i][j] = relevance of documents[j] to queries[i].
        Scores are cosine similarities in range [-1, 1] (typically [0, 1] for
        positive-relevance embeddings).
    """
    resolved_model = model_name or EMBED_MODEL
    if isinstance(queries, str):
        queries = [queries]

    logger.info(
        f"evaluate_relevance: {len(queries)} queries × {len(documents)} docs, "
        f"model={resolved_model}, task='{task_description[:60]}...'"
    )

    # Format queries with instruction template
    formatted_queries = [get_detailed_instruct(task_description, q) for q in queries]

    # Embed queries and documents separately so the instruction prefix
    # only applies to queries (matching MLX reference behavior)
    logger.debug("Embedding formatted queries...")
    query_embeddings = embed(
        formatted_queries,
        model=resolved_model,
        return_format="numpy",
        batch_size=batch_size,
        show_progress=show_progress,
        progress_description="Embedding queries",
    )

    logger.debug("Embedding documents...")
    doc_embeddings = embed(
        documents,
        model=resolved_model,
        return_format="numpy",
        batch_size=batch_size,
        show_progress=show_progress,
        progress_description="Embedding documents",
    )

    # Compute all-pairs cosine similarity (equivalent to normalized dot product)
    # Shape: (num_queries, num_documents)
    logger.debug("Computing similarity scores...")
    score_matrix = np.zeros((len(queries), len(documents)), dtype=np.float32)

    query_iter = tqdm(
        enumerate(query_embeddings),
        desc="Scoring",
        total=len(queries),
        disable=not show_progress,
    )
    for qi, q_emb in query_iter:
        for di, d_emb in enumerate(doc_embeddings):
            score_matrix[qi, di] = cosine_similarity(q_emb, d_emb)

    scores = score_matrix.tolist()

    logger.info(
        f"evaluate_relevance complete: returned {len(scores)}×"
        f"{len(scores[0]) if scores else 0} score matrix"
    )
    return scores


if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()

    task = "Given a web search query, retrieve relevant passages that answer the query"
    test_queries = [
        "What is the capital of China?",
        "Explain gravity",
    ]
    test_documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement of planets around the sun.",
    ]

    result_scores = evaluate_relevance(
        test_queries,
        test_documents,
        task,
        show_progress=True,
    )

    # Display results sorted by score (descending) per query
    console.print("\n[bold green]Relevance Evaluation Results[/bold green]")
    table = Table(show_header=True, header_style="bold magenta", show_lines=True)
    table.add_column("Rank", justify="center", style="dim", width=4)
    table.add_column("Query", style="cyan", no_wrap=False, max_width=40)
    table.add_column("Document", style="white", no_wrap=False, max_width=60)
    table.add_column("Score", justify="right", style="yellow")

    for qi, query in enumerate(test_queries):
        # Pair documents with scores and sort descending
        scored_docs = list(zip(result_scores[qi], test_documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        for rank, (score, doc) in enumerate(scored_docs, start=1):
            # Color-code scores for quick visual assessment
            if score >= 0.7:
                score_style = "bold green"
            elif score >= 0.4:
                score_style = "yellow"
            else:
                score_style = "dim red"

            table.add_row(
                str(rank),
                query if rank == 1 else "",  # Only show query on first row
                doc[:100] + ("..." if len(doc) > 100 else ""),
                f"[{score_style}]{score:.4f}[/{score_style}]",
            )

        # Add a separator between query groups for readability
        if qi < len(test_queries) - 1:
            table.add_row("", "", "", "")

    console.print(table)
