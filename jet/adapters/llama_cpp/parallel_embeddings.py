import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Union

import numpy as np
from openai import OpenAI
from rich.console import Console
from tqdm import tqdm  # for optional per-request progress feel if needed

console = Console()

# === CONFIG ===
SERVER_URL = os.getenv("LLAMA_CPP_EMBED_URL")
MODEL_NAME = os.getenv("LLAMA_CPP_EMBED_MODEL")

client = OpenAI(
    base_url=SERVER_URL,
    api_key="not-needed-for-local",  # llama.cpp ignores this
)


def embed_single(text: str, model: str = MODEL_NAME) -> list[float]:
    """Embed one text string via /v1/embeddings endpoint."""
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding


def embed_batch(
    texts: list[str],
    max_workers: int = 4,
    show_progress: bool = True,
    return_format: Literal["numpy", "list"] = "numpy",
) -> Union[list[list[float]], np.ndarray]:
    """
    Embed multiple texts in parallel using ThreadPoolExecutor.

    Args:
        texts: List of strings to embed.
        max_workers: Number of parallel threads.
        show_progress: Whether to show a progress bar.
        return_format: "numpy" to return np.ndarray, "list" for Python list.

    Returns:
        Embeddings in same order as input texts, either as list or np.ndarray.
    """
    embeddings: list[list[float] | None] = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(embed_single, text): i for i, text in enumerate(texts)
        }

        if show_progress:
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(texts),
                desc="Embedding documents",
                unit="doc",
            ):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    console.print(f"[red]Error embedding doc {idx}: {e}[/red]")
        else:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                embeddings[idx] = future.result()

    # Ensure no None remains
    embeddings = [e for e in embeddings if e is not None]

    if return_format == "numpy":
        return np.array(embeddings, dtype=np.float32)
    return embeddings
