import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Union

import numpy as np
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from openai import OpenAI
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn

console = Console()

# === CONFIG ===
SERVER_URL = os.getenv("LLAMA_CPP_EMBED_URL")
MODEL_NAME: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL")

client = OpenAI(
    base_url=SERVER_URL,
    api_key="not-needed-for-local",  # llama.cpp ignores this
)


def embed_single(
    text: str,
    model: LLAMACPP_EMBED_KEYS = MODEL_NAME,
    return_format: Literal["numpy", "list"] = "numpy",
) -> Union[list[float], np.ndarray]:
    """Embed one text string via /v1/embeddings endpoint.

    Args:
        text: Input text to embed
        model: Model identifier
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
    texts: list[str], model: LLAMACPP_EMBED_KEYS = MODEL_NAME
) -> list[list[float]]:
    """Embed a list of texts sequentially, returns list of embeddings in same order."""
    # Keep returning plain Python lists here (cheaper + conversion happens once in embed_batch)
    return [embed_single(t, model=model, return_format="list") for t in texts]


def embed_batch(
    texts: list[str],
    max_workers: int = 4,
    show_progress: bool = True,
    return_format: Literal["numpy", "list"] = "numpy",
    batch_size: int | None = 32,  # sensible default
    progress_description: str = "Embedding texts",
) -> Union[list[list[float]], np.ndarray]:
    """
    Embed multiple texts in parallel using ThreadPoolExecutor + batching.
    """
    if not texts:
        return np.array([]) if return_format == "numpy" else []

    if batch_size is None or batch_size <= 1:
        batch_size = 1

    total_texts = len(texts)

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
        task_id = progress.add_task(progress_description, total=total_texts)

    embeddings: list[list[float] | None] = [None] * total_texts

    batches = [
        (i, texts[i : i + batch_size]) for i in range(0, total_texts, batch_size)
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_info = {
            executor.submit(embed_chunk, batch_texts): (start_idx, len(batch_texts))
            for start_idx, batch_texts in batches
        }

        for future in as_completed(future_to_info):
            start_idx, batch_len = future_to_info[future]
            try:
                batch_emb = future.result()
                for offset, emb in enumerate(batch_emb):
                    embeddings[start_idx + offset] = emb

                # Update progress
                if show_progress and task_id is not None:
                    progress.update(task_id, advance=batch_len)

            except Exception as e:
                console.print(
                    f"[red]Error in batch starting at index {start_idx} "
                    f"({batch_len} texts): {e}[/red]"
                )
                # Optionally continue or raise

    if show_progress and progress is not None:
        progress.stop()

    # Filter just in case (shouldn't be needed)
    embeddings = [e for e in embeddings if e is not None]

    if len(embeddings) != total_texts:
        console.print(
            f"[yellow]Warning: Only {len(embeddings)}/{total_texts} texts embedded[/yellow]"
        )

    if return_format == "numpy":
        return np.array(embeddings, dtype=np.float32)
    return embeddings
