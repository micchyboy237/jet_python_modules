import pytest
from typing import List
from jet.models.embeddings.utils import calculate_dynamic_batch_size

@pytest.fixture
def cleanup():
    yield
    # No specific cleanup needed for this test

class TestCalculateDynamicBatchSize:
    def test_small_list_moderate_tokens(self, cleanup):
        """
        Given a small list of input texts with moderate token counts,
        When calculating the dynamic batch size,
        Then it should return a batch size that fits within memory limits.
        """
        token_counts: List[int] = [100, 200, 150]  # Max 200 tokens
        embedding_size: int = 768
        context_size: int = 512
        # Memory per input: 200 * 768 * 4 * 1.5 = 921,600 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 921,600, 4GB / 921,600) ≈ 4,340
        # Constrained by input length (3) and cap (128)
        expected_batch_size: int = 3

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_high_tokens(self, cleanup):
        """
        Given a large list of input texts with high token counts,
        When calculating the dynamic batch size,
        Then it should cap at 128 to prevent memory saturation.
        """
        token_counts: List[int] = [1000] * 1000  # Max 1000 tokens
        embedding_size: int = 1024
        context_size: int = 2048
        # Memory per input: 1000 * 1024 * 4 * 1.5 = 6,144,000 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 6,144,000, 4GB / 6,144,000) ≈ 652
        # Constrained by cap (128)
        expected_batch_size: int = 128

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_empty_list(self, cleanup):
        """
        Given an empty list of token counts,
        When calculating the dynamic batch size,
        Then it should return 1 to handle edge case gracefully.
        """
        token_counts: List[int] = []
        embedding_size: int = 768
        context_size: int = 512
        # Memory per input: 512 * 768 * 4 * 1.5 = 2,359,296 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 2,359,296, 4GB / 2,359,296) ≈ 1,695
        # Constrained by input length (0), so returns 1
        expected_batch_size: int = 1

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_variable_tokens_small_list(self, cleanup):
        """
        Given a small list with highly variable token counts,
        When calculating the dynamic batch size,
        Then it should use the maximum token count to avoid memory overflow.
        """
        token_counts: List[int] = [10, 1000, 50]  # Max 1000 tokens
        embedding_size: int = 768
        context_size: int = 512
        # Memory per input: 1000 * 768 * 4 * 1.5 = 4,608,000 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 4,608,000, 4GB / 4,608,000) ≈ 868
        # Constrained by input length (3) and cap (128)
        expected_batch_size: int = 3

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_small_embedding(self, cleanup):
        """
        Given a large list with a small embedding size,
        When calculating the dynamic batch size,
        Then it should allow a larger batch size within memory limits.
        """
        token_counts: List[int] = [200] * 100  # Max 200 tokens
        embedding_size: int = 128
        context_size: int = 512
        # Memory per input: 200 * 128 * 4 * 1.5 = 153,600 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 153,600, 4GB / 153,600) ≈ 26,041
        # Constrained by input length (100) and cap (128)
        expected_batch_size: int = 100

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_small_list_large_context(self, cleanup):
        """
        Given a small list with a large context size and small token counts,
        When calculating the dynamic batch size,
        Then it should use the maximum token count, not context size.
        """
        token_counts: List[int] = [50] * 20  # Max 50 tokens
        embedding_size: int = 768
        context_size: int = 8192
        # Memory per input: 50 * 768 * 4 * 1.5 = 230,400 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 230,400, 4GB / 230,400) ≈ 17,361
        # Constrained by input length (20) and cap (128)
        expected_batch_size: int = 20

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_low_memory_ratio(self, cleanup):
        """
        Given a large list with a low max_memory_usage ratio,
        When calculating the dynamic batch size,
        Then it should return a smaller batch size to respect the memory constraint.
        """
        token_counts: List[int] = [200] * 100  # Max 200 tokens
        embedding_size: int = 768
        context_size: int = 512
        max_memory_usage: float = 0.2
        # Memory per input: 200 * 768 * 4 * 1.5 = 921,600 bytes
        # RAM target: 8GB * 0.2 = 1.6GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(1.6GB / 921,600, 4GB / 921,600) ≈ 1,736
        # Constrained by input length (100) and cap (128)
        expected_batch_size: int = 100

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, max_memory_usage)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_average_500_tokens(self, cleanup):
        """
        Given a large list with an average of 500 tokens,
        When calculating the dynamic batch size,
        Then it should return a batch size that fits within memory limits.
        """
        token_counts: List[int] = [500] * 200  # Max 500 tokens (average 500)
        embedding_size: int = 768
        context_size: int = 1024
        # Memory per input: 500 * 768 * 4 * 1.5 = 2,304,000 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 2,304,000, 4GB / 2,304,000) ≈ 1,736
        # Constrained by input length (200) and cap (128)
        expected_batch_size: int = 128

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_average_2000_tokens(self, cleanup):
        """
        Given a large list with an average of 2000 tokens,
        When calculating the dynamic batch size,
        Then it should return a smaller batch size to respect memory constraints.
        """
        token_counts: List[int] = [2000] * 200  # Max 2000 tokens (average 2000)
        embedding_size: int = 768
        context_size: int = 4096
        # Memory per input: 2000 * 768 * 4 * 1.5 = 9,216,000 bytes
        # RAM target: 8GB * 0.8 = 6.4GB
        # VRAM target: 5GB * 0.8 = 4GB
        # Max batch size: min(6.4GB / 9,216,000, 4GB / 9,216,000) ≈ 434
        # Constrained by input length (200) and cap (128)
        expected_batch_size: int = 128

        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size)

        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"
