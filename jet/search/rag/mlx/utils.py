from typing import List, Optional
from collections import defaultdict
import numpy as np
import mlx.core as mx
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import psutil


def get_optimal_thread_workers(chunk_count: int, avg_tokens: float) -> int:
    """Determine the optimal number of thread workers for tokenization based on chunk count and token length.

    Args:
        chunk_count: Number of chunks to process.
        avg_tokens: Average number of tokens per chunk.

    Returns:
        Number of thread workers (1 for no threading, >1 for parallel execution).
    """
    # Constants
    MIN_CHUNKS_FOR_THREADING = 50  # No threading for small datasets
    MAX_WORKERS = min(os.cpu_count() or 8, 8)  # Cap at CPU cores (8 on M1)
    MEMORY_SAFETY_FACTOR = 0.8  # Use up to 80% of available memory
    TOKENS_PER_MB = 10000  # Rough estimate: 10K tokens ~ 1MB memory
    THREADING_OVERHEAD_MS = 100  # Estimated threading overhead in milliseconds
    # Estimated tokenization time per token (ms)
    TOKENIZATION_MS_PER_TOKEN = 0.01

    # Disable threading for small datasets
    if chunk_count < MIN_CHUNKS_FOR_THREADING:
        logger.debug(
            f"Chunk count {chunk_count} < {MIN_CHUNKS_FOR_THREADING}, using 1 worker")
        return 1

    # Estimate tokenization time to check if threading is worthwhile
    estimated_time_ms = chunk_count * avg_tokens * TOKENIZATION_MS_PER_TOKEN
    if estimated_time_ms < THREADING_OVERHEAD_MS:
        logger.debug(
            f"Estimated tokenization time {estimated_time_ms:.2f}ms < {THREADING_OVERHEAD_MS}ms, using 1 worker")
        return 1

    # Estimate memory usage
    estimated_memory_mb = (chunk_count * avg_tokens) / TOKENS_PER_MB
    available_memory_mb = psutil.virtual_memory().available / (1024 ** 2) * \
        MEMORY_SAFETY_FACTOR
    if estimated_memory_mb > available_memory_mb:
        workers = max(1, int(available_memory_mb /
                      (estimated_memory_mb / MAX_WORKERS)))
        logger.debug(
            f"Memory constraint: estimated {estimated_memory_mb:.2f}MB > available {available_memory_mb:.2f}MB, using {workers} workers")
        return workers

    # Scale workers based on chunk count and token length
    base_workers = min(MAX_WORKERS, int(
        np.log2(chunk_count / MIN_CHUNKS_FOR_THREADING) + 1))
    # Reduce workers for long chunks
    token_factor = max(0.5, min(1.0, 512 / avg_tokens))
    workers = max(1, int(base_workers * token_factor))
    logger.debug(
        f"Computed {workers} workers: base={base_workers}, token_factor={token_factor:.2f}, chunks={chunk_count}, avg_tokens={avg_tokens:.2f}")
    return workers
