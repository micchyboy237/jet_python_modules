import pytest
from typing import List
from jet.models.embeddings.utils import calculate_dynamic_batch_size

@pytest.fixture
def cleanup():
    yield

class TestCalculateDynamicBatchSize:
    def test_empty_list(self, cleanup):
        """
        Given an empty list of token counts,
        When calculating the dynamic batch size,
        Then it should return the minimum batch size to handle the edge case gracefully.
        """
        # Formula:
        # - token_counts = [], so max_tokens = context_size = 512
        # - bytes_per_token = embedding_size * 4 = 768 * 4 = 3072
        # - memory_per_input = max_tokens * bytes_per_token * memory_overhead_factor = 512 * 3072 * 1.5 = 2,359,296
        # - ram_limit = 16GB = 16 * 1024^3 = 17,179,869,184 (use_dynamic_limits=False)
        # - vram_limit = 6GB = 6 * 1024^3 = 6,442,450,944
        # - ram_target = ram_limit * max_memory_usage = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = vram_limit * max_vram_usage = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(ram_target // memory_per_input)) = max(1, int(13,743,895,347 // 2,359,296)) = 5827
        # - vram_batch_size = max(1, int(vram_target // memory_per_input)) = max(1, int(5,153,960,755 // 2,359,296)) = 2185
        # - optimal_batch_size = min(ram_batch_size, vram_batch_size) = min(5827, 2185) = 2185
        # - max_tokens = 512 (not < 10), so no small-token adjustment
        # - result = max(min_batch_size, min(optimal_batch_size, max_batch_size_cap, len(token_counts)))
        #          = max(8, min(2185, 512, 0)) = 8 (since len(token_counts) = 0)
        token_counts: List[int] = []
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 8
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_single_input_small_tokens(self, cleanup):
        """
        Given a single input with a small token count,
        When calculating the dynamic batch size,
        Then it should return the minimum batch size since the list has one element.
        """
        # Formula:
        # - token_counts = [5], so max_tokens = 5
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 5 * 3072 * 1.5 = 23,040
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 23,040)) = 596,518
        # - vram_batch_size = max(1, int(5,153,960,755 // 23,040)) = 223,695
        # - optimal_batch_size = min(596,518, 223,695) = 223,695
        # - max_tokens = 5 (< 10), so optimal_batch_size = min(223,695 * 2, 512) = 512
        # - result = max(8, min(512, 512, 1)) = 8 (since len(token_counts) = 1)
        token_counts: List[int] = [5]
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 8
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_small_list_moderate_tokens(self, cleanup):
        """
        Given a small list of input texts with moderate token counts,
        When calculating the dynamic batch size,
        Then it should return the list size since it’s smaller than the calculated batch size.
        """
        # Formula:
        # - token_counts = [100, 200, 150], so max_tokens = 200
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 200 * 3072 * 1.5 = 921,600
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 921,600)) = 14,908
        # - vram_batch_size = max(1, int(5,153,960,755 // 921,600)) = 5,590
        # - optimal_batch_size = min(14,908, 5,590) = 5,590
        # - max_tokens = 200 (not < 10), so no small-token adjustment
        # - result = max(8, min(5,590, 512, 3)) = 8 (since len(token_counts) = 3)
        token_counts: List[int] = [100, 200, 150]
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 8
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_ngrams_small_tokens_static(self, cleanup):
        """
        Given a large list of n-grams with small token counts and static memory limits,
        When calculating the dynamic batch size with use_dynamic_limits=False,
        Then it should return a larger batch size to optimize throughput.
        """
        # Formula:
        # - token_counts = [5] * 1000, so max_tokens = 5
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 5 * 3072 * 1.5 = 23,040
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 23,040)) = 596,518
        # - vram_batch_size = max(1, int(5,153,960,755 // 23,040)) = 223,695
        # - optimal_batch_size = min(596,518, 223,695) = 223,695
        # - max_tokens = 5 (< 10), so optimal_batch_size = min(223,695 * 2, 512) = 512
        # - result = max(8, min(512, 512, 1000)) = 512
        token_counts: List[int] = [5] * 1000
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 512
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_ngrams_small_tokens_dynamic(self, cleanup):
        """
        Given a large list of n-grams with small token counts and dynamic memory limits,
        When calculating the dynamic batch size with use_dynamic_limits=True,
        Then it should return a larger batch size to optimize throughput, assuming sufficient RAM.
        """
        # Formula (assuming psutil returns ~12GB available RAM for dynamic case):
        # - token_counts = [5] * 1000, so max_tokens = 5
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 5 * 3072 * 1.5 = 23,040
        # - ram_limit = 12 * 1024^3 = 12,884,901,888 (dynamic)
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 12,884,901,888 * 0.8 = 10,307,921,510
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(10,307,921,510 // 23,040)) = 447,391
        # - vram_batch_size = max(1, int(5,153,960,755 // 23,040)) = 223,695
        # - optimal_batch_size = min(447,391, 223,695) = 223,695
        # - max_tokens = 5 (< 10), so optimal_batch_size = min(223,695 * 2, 512) = 512
        # - result = max(8, min(512, 512, 1000)) = 512
        token_counts: List[int] = [5] * 1000
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 512
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=True)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_high_tokens(self, cleanup):
        """
        Given a large list of input texts with high token counts,
        When calculating the dynamic batch size,
        Then it should return a smaller batch size to prevent memory saturation.
        """
        # Formula:
        # - token_counts = [1000] * 1000, so max_tokens = 1000
        # - bytes_per_token = 1024 * 4 = 4096
        # - memory_per_input = 1000 * 4096 * 1.5 = 6,144,000
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 6,144,000)) = 2236
        # - vram_batch_size = max(1, int(5,153,960,755 // 6,144,000)) = 838
        # - optimal_batch_size = min(2236, 838) = 838
        # - max_tokens = 1000 (not < 10), so no small-token adjustment
        # - result = max(8, min(838, 512, 1000)) = 512
        token_counts: List[int] = [1000] * 1000
        embedding_size: int = 1024
        context_size: int = 2048
        expected_batch_size: int = 512
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_variable_tokens_small_list(self, cleanup):
        """
        Given a small list with highly variable token counts,
        When calculating the dynamic batch size,
        Then it should use the maximum token count to avoid memory overflow.
        """
        # Formula:
        # - token_counts = [10, 1000, 50], so max_tokens = 1000
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 1000 * 3072 * 1.5 = 4,608,000
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 4,608,000)) = 2981
        # - vram_batch_size = max(1, int(5,153,960,755 // 4,608,000)) = 1118
        # - optimal_batch_size = min(2981, 1118) = 1118
        # - max_tokens = 1000 (not < 10), so no small-token adjustment
        # - result = max(8, min(1118, 512, 3)) = 8 (since len(token_counts) = 3)
        token_counts: List[int] = [10, 1000, 50]
        embedding_size: int = 768
        context_size: int = 512
        expected_batch_size: int = 8
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_small_embedding(self, cleanup):
        """
        Given a large list with a small embedding size,
        When calculating the dynamic batch size,
        Then it should allow a larger batch size within memory limits.
        """
        # Formula:
        # - token_counts = [200] * 100, so max_tokens = 200
        # - bytes_per_token = 128 * 4 = 512
        # - memory_per_input = 200 * 512 * 1.5 = 153,600
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 153,600)) = 89,505
        # - vram_batch_size = max(1, int(5,153,960,755 // 153,600)) = 33,552
        # - optimal_batch_size = min(89,505, 33,552) = 33,552
        # - max_tokens = 200 (not < 10), so no small-token adjustment
        # - result = max(8, min(33,552, 512, 100)) = 100 (since len(token_counts) = 100)
        token_counts: List[int] = [200] * 100
        embedding_size: int = 128
        context_size: int = 512
        expected_batch_size: int = 100
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_small_list_large_context(self, cleanup):
        """
        Given a small list with a large context size and small token counts,
        When calculating the dynamic batch size,
        Then it should use the maximum token count, not context size.
        """
        # Formula:
        # - token_counts = [50] * 20, so max_tokens = 50
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 50 * 3072 * 1.5 = 230,400
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 230,400)) = 59,651
        # - vram_batch_size = max(1, int(5,153,960,755 // 230,400)) = 22,369
        # - optimal_batch_size = min(59,651, 22,369) = 22,369
        # - max_tokens = 50 (not < 10), so no small-token adjustment
        # - result = max(8, min(22,369, 512, 20)) = 20 (since len(token_counts) = 20)
        token_counts: List[int] = [50] * 20
        embedding_size: int = 768
        context_size: int = 8192
        expected_batch_size: int = 20
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_low_memory_ratio(self, cleanup):
        """
        Given a large list with a low max_memory_usage ratio,
        When calculating the dynamic batch size,
        Then it should return a smaller batch size to respect the memory constraint.
        """
        # Formula:
        # - token_counts = [200] * 100, so max_tokens = 200
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 200 * 3072 * 1.5 = 921,600
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.2 = 3,435,973,836
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(3,435,973,836 // 921,600)) = 3727
        # - vram_batch_size = max(1, int(5,153,960,755 // 921,600)) = 5590
        # - optimal_batch_size = min(3727, 5590) = 3727
        # - max_tokens = 200 (not < 10), so no small-token adjustment
        # - result = max(8, min(3727, 512, 100)) = 100 (since len(token_counts) = 100)
        token_counts: List[int] = [200] * 100
        embedding_size: int = 768
        context_size: int = 512
        max_memory_usage: float = 0.2
        expected_batch_size: int = 100
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, max_memory_usage=max_memory_usage, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_average_500_tokens(self, cleanup):
        """
        Given a large list with an average of 500 tokens,
        When calculating the dynamic batch size,
        Then it should return a batch size that fits within memory limits.
        """
        # Formula:
        # - token_counts = [500] * 200, so max_tokens = 500
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 500 * 3072 * 1.5 = 2,304,000
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 2,304,000)) = 5963
        # - vram_batch_size = max(1, int(5,153,960,755 // 2,304,000)) = 2236
        # - optimal_batch_size = min(5963, 2236) = 2236
        # - max_tokens = 500 (not < 10), so no small-token adjustment
        # - result = max(8, min(2236, 512, 200)) = 200 (since len(token_counts) = 200)
        token_counts: List[int] = [500] * 200
        embedding_size: int = 768
        context_size: int = 1024
        expected_batch_size: int = 200
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"

    def test_large_list_average_2000_tokens(self, cleanup):
        """
        Given a large list with an average of 2000 tokens,
        When calculating the dynamic batch size,
        Then it should return a smaller batch size to respect memory constraints.
        """
        # Formula:
        # - token_counts = [2000] * 200, so max_tokens = 2000
        # - bytes_per_token = 768 * 4 = 3072
        # - memory_per_input = 2000 * 3072 * 1.5 = 9,216,000
        # - ram_limit = 16 * 1024^3 = 17,179,869,184
        # - vram_limit = 6 * 1024^3 = 6,442,450,944
        # - ram_target = 17,179,869,184 * 0.8 = 13,743,895,347
        # - vram_target = 6,442,450,944 * 0.8 = 5,153,960,755
        # - ram_batch_size = max(1, int(13,743,895,347 // 9,216,000)) = 1490
        # - vram_batch_size = max(1, int(5,153,960,755 // 9,216,000)) = 559
        # - optimal_batch_size = min(1490, 559) = 559
        # - max_tokens = 2000 (not < 10), so no small-token adjustment
        # - result = max(8, min(559, 512, 200)) = 200 (since len(token_counts) = 200)
        token_counts: List[int] = [2000] * 200
        embedding_size: int = 768
        context_size: int = 4096
        expected_batch_size: int = 200
        result = calculate_dynamic_batch_size(token_counts, embedding_size, context_size, use_dynamic_limits=False)
        assert result == expected_batch_size, f"Expected batch size {expected_batch_size}, got {result}"
