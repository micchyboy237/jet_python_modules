import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Union, overload

import numpy as np
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

console = Console()

# === CONFIG ===
SERVER_URL = os.getenv("LLAMA_CPP_EMBED_URL")
MODEL_NAME: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL")
DEFAULT_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "")
DEFAULT_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "")

client = OpenAI(
    base_url=SERVER_URL,
    api_key="not-needed-for-local",
)


def _resolve_prefix(prefix: str | None, default: str) -> str | None:
    """Resolve prefix: explicit arg > env default. Returns None if both are empty."""
    if prefix is not None:
        return prefix if prefix else None
    return default if default else None


@overload
def embed(
    text: str,
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    return_format: Literal["numpy", "list"] = "numpy",
    prefix: str | None = None,
) -> Union[list[float], np.ndarray]: ...


@overload
def embed(
    text: list[str],
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    return_format: Literal["numpy", "list"] = "numpy",
    max_workers: int = 6,
    show_progress: bool = True,
    batch_size: int | None = 32,
    progress_description: str = "Embedding texts",
    prefix: str | None = None,
) -> Union[list[list[float]], np.ndarray]: ...


def embed(
    text: Union[str, list[str]],
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    return_format: Literal["numpy", "list"] = "numpy",
    max_workers: int = 6,
    show_progress: bool = True,
    batch_size: int | None = 32,
    progress_description: str = "Embedding texts",
    prefix: str | None = None,
) -> Union[list[float], list[list[float]], np.ndarray]:
    """
    Unified embedding interface.

    - str input → uses embed_single
    - list[str] input → uses embed_batch

    Prefix behavior:
      - If `prefix` is provided, it's used directly (empty string = no prefix).
      - If `prefix` is None, no prefix is applied here — callers like vector_search
        resolve env vars and pass an explicit prefix.

    Keeps API ergonomic while reusing existing implementations.
    """
    if isinstance(text, str):
        return embed_single(
            text=text,
            model=model,
            return_format=return_format,
            prefix=prefix,
        )

    if isinstance(text, list):
        return embed_batch(
            texts=text,
            model=model,
            max_workers=max_workers,
            show_progress=show_progress,
            return_format=return_format,
            batch_size=batch_size,
            progress_description=progress_description,
            prefix=prefix,
        )

    raise TypeError(f"Unsupported input type: {type(text)}")


def embed_single(
    text: str,
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    return_format: Literal["numpy", "list"] = "numpy",
    prefix: str | None = None,
) -> Union[list[float], np.ndarray]:
    """Embed one text string via /v1/embeddings endpoint.

    Args:
        text: Input text to embed.
        model: Model identifier.
        return_format: "numpy" returns np.ndarray (default), "list" returns Python list.
        prefix: Optional prefix prepended to text. Falls back to DEFAULT_QUERY_PREFIX env var.

    Returns:
        Embedding vector as numpy array (default) or Python list.
    """
    resolved_prefix = _resolve_prefix(prefix, DEFAULT_QUERY_PREFIX)
    if resolved_prefix:
        text = f"{resolved_prefix}{text}"

    response = client.embeddings.create(
        input=text,
        model=model,
    )
    embedding = response.data[0].embedding

    if return_format == "list":
        return embedding
    return np.array(embedding, dtype=np.float32)


def embed_chunk(
    texts: list[str],
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    prefix: str | None = None,
) -> list[list[float]]:
    """Embed a list of texts sequentially, returns list of embeddings in same order.

    Args:
        texts: List of texts to embed.
        model: Model identifier.
        prefix: Optional prefix prepended to each text. Falls back to DEFAULT_DOC_PREFIX env var.
    """
    resolved_prefix = _resolve_prefix(prefix, DEFAULT_DOC_PREFIX)
    return [
        embed_single(t, model=model, return_format="list", prefix=resolved_prefix)
        for t in texts
    ]


def embed_batch(
    texts: list[str],
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    max_workers: int = 6,
    show_progress: bool = True,
    return_format: Literal["numpy", "list"] = "numpy",
    batch_size: int | None = 32,
    progress_description: str = "Embedding texts",
    prefix: str | None = None,
) -> Union[list[list[float]], np.ndarray]:
    """
    Embed multiple texts in parallel using ThreadPoolExecutor + batching.
    Deduplicates input texts for efficiency, reconstructs output list in original order.

    Args:
        texts: List of texts to embed.
        model: Model identifier.
        max_workers: Max parallel threads.
        show_progress: Whether to display a progress bar.
        return_format: "numpy" returns np.ndarray, "list" returns Python list.
        batch_size: Texts per batch. None or <=1 disables batching.
        progress_description: Label for the progress bar.
        prefix: Optional prefix prepended to each text. Falls back to DEFAULT_DOC_PREFIX env var.
    """
    resolved_prefix = _resolve_prefix(prefix, DEFAULT_DOC_PREFIX)

    if not texts:
        return np.array([]) if return_format == "numpy" else []

    # ── Deduplicate while preserving original index mapping ───────────────
    text_to_indices: dict[str, list[int]] = {}
    for idx, text in enumerate(texts):
        text_to_indices.setdefault(text, []).append(idx)

    unique_texts = list(text_to_indices.keys())

    total_unique = len(unique_texts)
    total_texts = len(texts)

    deduped_count = total_texts - total_unique
    if deduped_count > 0:
        console.print(
            f"[yellow]Deduped: {total_texts} → {total_unique} (removed {deduped_count} duplicates)[/yellow]"
        )

    if batch_size is None or batch_size <= 1:
        batch_size = 1

    # ── Progress setup ────────────────────────────────
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

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(embed_chunk, batch_texts, model, resolved_prefix): (
                start_idx,
                len(batch_texts),
            )
            for start_idx, batch_texts in batches
        }

        for future in as_completed(future_to_info):
            start_idx, batch_len = future_to_info[future]
            try:
                batch_emb = future.result()

                for offset, emb in enumerate(batch_emb):
                    unique_text = unique_texts[start_idx + offset]
                    for original_idx in text_to_indices[unique_text]:
                        embeddings[original_idx] = emb

                if show_progress and task_id is not None:
                    progress.update(task_id, advance=batch_len)

            except Exception as e:
                console.print(
                    f"[red]Error in batch starting at index {start_idx} "
                    f"({batch_len} texts): {e}[/red]"
                )

    if show_progress and progress is not None:
        progress.stop()

    embeddings = [e for e in embeddings if e is not None]

    if len(embeddings) != total_texts:
        console.print(
            f"[yellow]Warning: Only {len(embeddings)}/{total_texts} texts embedded[/yellow]"
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
