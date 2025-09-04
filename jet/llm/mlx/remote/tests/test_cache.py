import pytest
import mlx.core as mx
from typing import List, Any

# Assuming the above functions and cache classes are in a module named `cache`
from mlx_lm.models.cache import (
    KVCache,
    QuantizedKVCache,
    RotatingKVCache,
    save_prompt_cache,
)
from jet.llm.mlx.cache import calculate_prompt_cache_memory_size, calculate_prompt_cache_disk_size


@pytest.fixture
def kv_cache():
    """Fixture to create a KVCache with realistic data."""
    cache = KVCache()
    # 1 batch, 8 heads, 256 tokens, 128 dim
    cache.keys = mx.zeros((1, 8, 256, 128), dtype=mx.float16)
    cache.values = mx.zeros((1, 8, 256, 128), dtype=mx.float16)
    cache.offset = 256
    return cache


@pytest.fixture
def quantized_kv_cache():
    """Fixture to create a QuantizedKVCache with realistic data."""
    cache = QuantizedKVCache(group_size=64, bits=8)
    cache.keys = (
        mx.zeros((1, 8, 256, 128 // 2), dtype=mx.uint32),  # Quantized data
        mx.zeros((1, 8, 256, 128 // 64), dtype=mx.float16),  # Scales
        mx.zeros((1, 8, 256, 128 // 64), dtype=mx.float16),  # Biases
    )
    cache.values = (
        mx.zeros((1, 8, 256, 128 // 2), dtype=mx.uint32),
        mx.zeros((1, 8, 256, 128 // 64), dtype=mx.float16),
        mx.zeros((1, 8, 256, 128 // 64), dtype=mx.float16),
    )
    cache.offset = 256
    # Step, offset, group_size, bits
    cache.meta_state = tuple(map(str, (256, 256, 64, 8)))
    return cache


@pytest.fixture
def rotating_kv_cache():
    """Fixture to create a RotatingKVCache with realistic data."""
    cache = RotatingKVCache(max_size=512, keep=4, step=256)
    # 1 batch, 8 heads, 256 tokens, 128 dim
    cache.keys = mx.zeros((1, 8, 256, 128), dtype=mx.float16)
    cache.values = mx.zeros((1, 8, 256, 128), dtype=mx.float16)
    cache.offset = 256
    cache._idx = 256
    # keep, max_size, step, offset, _idx
    cache.meta_state = tuple(map(str, (4, 512, 256, 256, 256)))
    return cache


class TestPromptCacheSize:
    def test_calculate_memory_size_kv_cache(self, kv_cache):
        """
        Test memory size calculation for a single KVCache.
        Given: A KVCache with float16 keys and values.
        When: Calculating the memory size.
        Then: The size matches the expected value based on array shapes.
        """
        # Given
        prompt_cache: List[Any] = [kv_cache]
        # Expected size:
        # keys: 1 * 8 * 256 * 128 * 2 bytes (float16) = 524,288 bytes
        # values: 1 * 8 * 256 * 128 * 2 bytes = 524,288 bytes
        # meta_state: "" (0 bytes)
        expected_size = 524288 + 524288

        # When
        result_size = calculate_prompt_cache_memory_size(prompt_cache)

        # Then
        assert result_size == expected_size, f"Expected {expected_size} bytes, got {result_size} bytes"

    def test_calculate_memory_size_quantized_kv_cache(self, quantized_kv_cache):
        """
        Test memory size calculation for a single QuantizedKVCache.
        Given: A QuantizedKVCache with uint32 data and float16 scales/biases.
        When: Calculating the memory size.
        Then: The size matches the expected value based on array shapes and meta_state.
        """
        # Given
        prompt_cache: List[Any] = [quantized_kv_cache]
        # Expected size:
        # keys:
        #   - data: 1 * 8 * 256 * (128 // 2) * 4 bytes (uint32) = 524,288 bytes
        #   - scales: 1 * 8 * 256 * (128 // 64) * 2 bytes (float16) = 8,192 bytes
        #   - biases: 1 * 8 * 256 * (128 // 64) * 2 bytes = 8,192 bytes
        # values: same as keys = 524,288 + 8,192 + 8,192 bytes
        # meta_state: tuple of strings ("256", "256", "64", "8") ≈ 3 + 3 + 2 + 1 = 9 bytes
        expected_size = (524288 + 8192 + 8192) * 2 + 9

        # When
        result_size = calculate_prompt_cache_memory_size(prompt_cache)

        # Then
        assert result_size == expected_size, f"Expected {expected_size} bytes, got {result_size} bytes"

    def test_calculate_memory_size_rotating_kv_cache(self, rotating_kv_cache):
        """
        Test memory size calculation for a single RotatingKVCache.
        Given: A RotatingKVCache with float16 keys and values.
        When: Calculating the memory size.
        Then: The size matches the expected value based on array shapes and meta_state.
        """
        # Given
        prompt_cache: List[Any] = [rotating_kv_cache]
        # Expected size:
        # keys: 1 * 8 * 256 * 128 * 2 bytes (float16) = 524,288 bytes
        # values: 1 * 8 * 256 * 128 * 2 bytes = 524,288 bytes
        # meta_state: tuple of strings ("4", "512", "256", "256", "256") ≈ 1 + 3 + 3 + 3 + 3 = 13 bytes
        expected_size = 524288 + 524288 + 13

        # When
        result_size = calculate_prompt_cache_memory_size(prompt_cache)

        # Then
        assert result_size == expected_size, f"Expected {expected_size} bytes, got {result_size} bytes"

    def test_calculate_disk_size_kv_cache(self, kv_cache, tmp_path):
        """
        Test disk size calculation for a single KVCache.
        Given: A KVCache saved to a .safetensors file.
        When: Calculating the disk size.
        Then: The size is greater than zero and reasonable.
        """
        # Given
        prompt_cache: List[Any] = [kv_cache]
        # Save to a temporary file to get actual disk size
        temp_file = tmp_path / "test.safetensors"
        save_prompt_cache(str(temp_file), prompt_cache)
        expected_size = temp_file.stat().st_size

        # When
        result_size = calculate_prompt_cache_disk_size(prompt_cache)

        # Then
        assert result_size == expected_size, f"Expected {expected_size} bytes, got {result_size} bytes"
        assert result_size > 0, "Disk size should be greater than zero"


@pytest.fixture(autouse=True)
def cleanup():
    """Clean up any temporary files after tests."""
    yield
    # No specific cleanup needed since calculate_prompt_cache_disk_size handles file deletion
