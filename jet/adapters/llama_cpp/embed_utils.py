import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Union, overload

import numpy as np
from jet.adapters.llama_cpp.config import EMBED_MODEL
from jet.adapters.llama_cpp.factory import get_embedding_client
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

console = Console()
client = get_embedding_client()


@overload
def embed(
    text: str,
    model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
    return_format: Literal["numpy", "list"] = "numpy",
    prefix: str | None = None,
) -> Union[list[float], np.ndarray]: ...


@overload
def embed(
    text: list[str],
    model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
    return_format: Literal["numpy", "list"] = "numpy",
    max_workers: int = 6,
    show_progress: bool = True,
    batch_size: int | None = 64,
    progress_description: str = "Embedding texts",
    prefix: str | None = None,
) -> Union[list[list[float]], np.ndarray]: ...


def embed(
    text: Union[str, list[str]],
    model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
    return_format: Literal["numpy", "list"] = "numpy",
    max_workers: int = 6,
    show_progress: bool = True,
    batch_size: int | None = 64,
    progress_description: str = "Embedding texts",
    prefix: str | None = None,
) -> Union[list[float], list[list[float]], np.ndarray]:
    """
    Unified embedding interface for local or remote llama.cpp servers.

    - str input → uses embed_single
    - list[str] input → uses embed_batch

    Remote Optimization Notes:
    - Default batch_size=64 amortizes network RTT over more texts per request
    - Default max_workers=2 prevents TCP connection saturation on WiFi/LAN
    - For wired gigabit connections, increase batch_size to 128 and max_workers to 4
    """
    if isinstance(text, str):
        if prefix:
            text = f"{prefix}{text}"
        return embed_single(
            text=text,
            model=model,
            return_format=return_format,
        )

    if isinstance(text, list):
        if prefix:
            text = [f"{prefix}{t}" for t in text]
        return embed_batch(
            texts=text,
            model=model,
            max_workers=max_workers,
            show_progress=show_progress,
            return_format=return_format,
            batch_size=batch_size,
            progress_description=progress_description,
        )

    raise TypeError(f"Unsupported input type: {type(text)}")


def embed_single(
    text: str,
    model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
    return_format: Literal["numpy", "list"] = "numpy",
) -> Union[list[float], np.ndarray]:
    """Embed one text string via /v1/embeddings endpoint.

    Args:
        text: Input text to embed
        model: Model identifier. Defaults to EMBED_MODEL from config.
        return_format: "numpy" returns np.ndarray (default), "list" returns Python list

    Returns:
        Embedding vector as numpy array (default) or Python list
    """
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    embedding = response.data[0].embedding

    if return_format == "list":
        return embedding
    return np.array(embedding, dtype=np.float32)


def embed_chunk(
    texts: list[str], model: LLAMACPP_EMBED_KEYS = EMBED_MODEL
) -> list[list[float]]:
    """Embed a list of texts sequentially, returns list of embeddings in same order."""
    return [embed_single(t, model=model, return_format="list") for t in texts]


def embed_batch(
    texts: list[str],
    model: LLAMACPP_EMBED_KEYS = EMBED_MODEL,
    max_workers: int = 6,
    show_progress: bool = True,
    return_format: Literal["numpy", "list"] = "numpy",
    batch_size: int | None = 64,
    progress_description: str = "Embedding texts",
    request_timeout: float = 120.0,
) -> Union[list[list[float]], np.ndarray]:
    """
    Embed multiple texts in parallel using ThreadPoolExecutor + batching.

    Optimized for remote llama.cpp servers (Mac M1 client → Windows PC server):
    - Larger default batch_size (64) amortizes network round-trip latency
    - Lower default max_workers (2) prevents TCP/WiFi saturation
    - Connectivity check before processing avoids silent hangs
    - Per-batch progress updates eliminate "stuck at 0%" perception
    - Request timeout prevents indefinite blocking on network failures

    Args:
        texts: List of text strings to embed
        model: Model identifier. Defaults to EMBED_MODEL from config.
        max_workers: Number of concurrent threads (default: 2 for remote safety)
        show_progress: Whether to show progress bar
        return_format: "numpy" or "list"
        batch_size: Number of texts per batch (default: 64 for remote RTT amortization)
        progress_description: Description for progress bar
        request_timeout: Seconds to wait per batch before raising TimeoutError

    Returns:
        Embeddings as numpy array or list of lists
    """
    if not texts:
        return np.array([]) if return_format == "numpy" else []

    # Quick connectivity check before starting expensive work
    if show_progress:
        console.print("[dim]Connecting to embedding server...[/dim]")
        start_time = time.monotonic()
        try:
            client.models.list()
            elapsed = time.monotonic() - start_time
            console.print(f"[green]Server reachable ({elapsed:.0f}ms RTT)[/green]")
        except Exception as e:
            console.print(
                f"[red]Server unreachable: {e}. Check network/server status.[/red]"
            )
            raise

    # Deduplicate input texts for efficiency
    text_to_indices: dict[str, list[int]] = {}
    for idx, text in enumerate(texts):
        text_to_indices.setdefault(text, []).append(idx)

    unique_texts = list(text_to_indices.keys())
    total_unique = len(unique_texts)
    total_texts = len(texts)
    deduped_count = total_texts - total_unique

    if deduped_count > 0:
        console.print(
            f"[yellow]Deduped: {total_texts} → {total_unique} "
            f"(removed {deduped_count} duplicates)[/yellow]"
        )

    if batch_size is None or batch_size <= 1:
        batch_size = 1

    # Initialize progress bar
    progress = None
    task_id: TaskID | None = None
    if show_progress:
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total} texts)"),
            transient=True,
        )
        progress.start()
        task_id = progress.add_task(progress_description, total=total_unique)

    embeddings: list[list[float] | None] = [None] * total_texts
    batches = [
        (i, unique_texts[i : i + batch_size])
        for i in range(0, total_unique, batch_size)
    ]

    # Cap workers to prevent network/GPU saturation
    actual_workers = min(max_workers, 6)
    if actual_workers < max_workers:
        console.print(
            f"[yellow]Limiting workers to {actual_workers} (requested {max_workers}) "
            f"to prevent network/GPU saturation[/yellow]"
        )

    with ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_info = {
            executor.submit(embed_chunk, batch_texts, model): (
                start_idx,
                len(batch_texts),
            )
            for start_idx, batch_texts in batches
        }

        completed_count = 0
        for future in as_completed(future_to_info):
            start_idx, batch_len = future_to_info[future]
            try:
                batch_emb = future.result(timeout=request_timeout)

                # Map embeddings back to original indices (handles duplicates)
                for offset, emb in enumerate(batch_emb):
                    unique_text = unique_texts[start_idx + offset]
                    for original_idx in text_to_indices[unique_text]:
                        embeddings[original_idx] = emb

                # Update progress immediately after each batch completes
                completed_count += batch_len
                if show_progress and task_id is not None:
                    progress.update(task_id, completed=completed_count)

            except TimeoutError:
                console.print(
                    f"[red]Timeout after {request_timeout}s for batch at index "
                    f"{start_idx}. Check server load/network.[/red]"
                )
            except Exception as e:
                console.print(
                    f"[red]Error in batch starting at index {start_idx} "
                    f"({batch_len} texts): {e}[/red]"
                )

    if show_progress and progress is not None:
        progress.stop()

    # Filter out any None entries from failed batches
    embeddings = [e for e in embeddings if e is not None]
    if len(embeddings) != total_texts:
        console.print(
            f"[yellow]Warning: Only {len(embeddings)}/{total_texts} "
            f"texts embedded successfully[/yellow]"
        )

    if return_format == "numpy":
        return np.array(embeddings, dtype=np.float32)
    return embeddings


if __name__ == "__main__":
    import numpy as np

    query = "What is a giant panda?"
    docs = [
        "The giant panda is a bear species endemic to China.",
        "Python is a high-level programming language.",
        "Bears are carnivoran mammals of the family Ursidae.",
        "Machine learning is a subset of artificial intelligence.",
        "Pandas eat bamboo and live in mountainous regions.",
    ]

    print("Embedding query and documents...")
    query_embedding = embed(query)
    doc_embeddings = embed(docs)
    print(f"Shape of query_embedding: {np.shape(query_embedding)}")
    print(f"Shape of doc_embeddings: {np.shape(doc_embeddings)}")
