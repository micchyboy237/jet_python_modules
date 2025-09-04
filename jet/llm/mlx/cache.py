import tempfile
import os
from jet.logger import logger
import mlx.core as mx
import numpy as np
from typing import List, Any
from mlx_lm.models.cache import (
    save_prompt_cache,
)


from typing import List, Any
import mlx.core as mx


def _get_mlx_dtype_size(dtype: mx.Dtype) -> int:
    """
    Get the size in bytes of an MLX data type.

    Args:
        dtype (mx.Dtype): The MLX data type.

    Returns:
        int: Size in bytes.

    Raises:
        ValueError: If the data type is not supported.
    """
    dtype_sizes = {
        mx.float16: 2,
        mx.float32: 4,
        mx.uint32: 4,
        mx.int32: 4,
        mx.int64: 8,
    }
    if dtype not in dtype_sizes:
        raise ValueError(f"Unsupported MLX data type: {dtype}")
    return dtype_sizes[dtype]


def calculate_prompt_cache_memory_size(prompt_cache: List[Any]) -> int:
    """
    Calculate the total memory size (in bytes) of a prompt cache.

    Args:
        prompt_cache (List[Any]): List of cache objects (e.g., KVCache, RotatingKVCache).

    Returns:
        int: Total memory size in bytes.
    """
    total_size = 0
    for cache in prompt_cache:
        # Get the state (keys, values, or quantized equivalents)
        state = cache.state
        if isinstance(state, (list, tuple)):
            for array in state:
                if isinstance(array, (list, tuple)):  # Handle QuantizedKVCache
                    for sub_array in array:
                        if isinstance(sub_array, mx.array):
                            total_size += sub_array.size * \
                                _get_mlx_dtype_size(sub_array.dtype)
                elif isinstance(array, mx.array):
                    total_size += array.size * _get_mlx_dtype_size(array.dtype)
        elif isinstance(state, mx.array):
            total_size += state.size * _get_mlx_dtype_size(state.dtype)

        # Include meta_state if it contributes to memory
        meta_state = cache.meta_state
        if isinstance(meta_state, (str, tuple)):
            if isinstance(meta_state, str):
                total_size += len(meta_state.encode('utf-8'))
            elif isinstance(meta_state, tuple):
                total_size += sum(len(str(item).encode('utf-8'))
                                  for item in meta_state)

    return total_size


def calculate_prompt_cache_disk_size(prompt_cache: List[Any], metadata: dict = None) -> int:
    """
    Calculate the disk size (in bytes) of a prompt cache when saved to a .safetensors file.

    Args:
        prompt_cache (List[Any]): List of cache objects (e.g., KVCache, RotatingKVCache).
        metadata (dict, optional): Metadata to include when saving. Defaults to None.

    Returns:
        int: Disk size in bytes.
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
        file_name = tmp_file.name
        # Save the prompt cache to the temporary file
        save_prompt_cache(file_name, prompt_cache, metadata or {})
        # Get the file size
        disk_size = os.path.getsize(file_name)

    # Clean up the temporary file
    os.unlink(file_name)
    return disk_size
