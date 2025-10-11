from typing import List

def calculate_dynamic_batch_size(
    token_counts: List[int],
    embedding_size: int,
    context_size: int,
    max_memory_usage: float = 0.8,
    max_vram_usage: float = 0.8,
    memory_overhead_factor: float = 1.5
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
    
    Returns:
        Optimal batch size as an integer.
    """
    # Use max token count for conservative memory estimation
    max_tokens = max(token_counts) if token_counts else context_size
    
    # Estimate memory per token (assume float32 for embeddings)
    bytes_per_token = embedding_size * 4  # 4 bytes per float32
    memory_per_input = max_tokens * bytes_per_token * memory_overhead_factor
    
    # Static memory limits for RAM (8GB) and VRAM (5GB)
    ram_limit = 8 * 1024 * 1024 * 1024  # 8GB in bytes
    vram_limit = 5 * 1024 * 1024 * 1024  # 5GB in bytes
    ram_target = ram_limit * max_memory_usage
    vram_target = vram_limit * max_vram_usage
    
    # Calculate max batch size based on memory constraints
    ram_batch_size = max(1, int(ram_target // memory_per_input))
    vram_batch_size = max(1, int(vram_target // memory_per_input))
    max_batch_size = min(ram_batch_size, vram_batch_size)
    
    # Constrain by reasonable defaults for your hardware (16GB RAM, GTX 1660)
    max_batch_size = min(max_batch_size, 128)  # Cap for stability
    return max(1, min(max_batch_size, len(token_counts)))  # Ensure at least 1, at most input count
