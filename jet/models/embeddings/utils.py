from typing import List
import psutil

def calculate_dynamic_batch_size(
    token_counts: List[int],
    embedding_size: int,
    context_size: int,
    max_memory_usage: float = 0.8,
    max_vram_usage: float = 0.8,
    memory_overhead_factor: float = 1.5,
    max_batch_size_cap: int = 128,
    min_batch_size: int = 8,
    use_dynamic_limits: bool = False
) -> int:
    """
    Calculate optimal batch size based on token counts, embedding size, and memory limits.
    Args:
        token_counts: List of token counts for each input.
        embedding_size: Size of each embedding vector.
        context_size: Maximum context size of the model.
        max_memory_usage: Fraction of available RAM to use (default: 0.8).
        max_vram_usage: Fraction of available VRAM to use (default: 0.8).
        memory_overhead_factor: Multiplier for model-specific memory overhead (default: 1.5).
        max_batch_size_cap: Maximum allowed batch size (default: 128).
        min_batch_size: Minimum batch size to reduce API overhead (default: 8).
        use_dynamic_limits: Use psutil for dynamic memory detection if True, else use static defaults (default: False).
    Returns:
        Optimal batch size as an integer.
    """
    # Use dynamic limits if enabled, else use static defaults based on PC specs (16GB RAM, 6GB VRAM)
    if use_dynamic_limits and hasattr(psutil, 'virtual_memory'):
        ram_limit = psutil.virtual_memory().available
    else:
        ram_limit = 16 * 1024 * 1024 * 1024  # 16GB
    vram_limit = 6 * 1024 * 1024 * 1024  # 6GB (GTX 1660)

    max_tokens = max(token_counts) if token_counts else context_size
    bytes_per_token = embedding_size * 4  # float32
    memory_per_input = max_tokens * bytes_per_token * memory_overhead_factor

    ram_target = ram_limit * max_memory_usage
    vram_target = vram_limit * max_vram_usage
    ram_batch_size = max(1, int(ram_target // memory_per_input))
    vram_batch_size = max(1, int(vram_target // memory_per_input))
    optimal_batch_size = min(ram_batch_size, vram_batch_size)

    # Adjust for small token counts to favor larger batches for throughput
    if max_tokens < 10:  # Typical for n-grams
        optimal_batch_size = min(optimal_batch_size * 2, max_batch_size_cap)

    # Ensure batch size is within bounds
    return max(min_batch_size, min(optimal_batch_size, max_batch_size_cap, len(token_counts)))
