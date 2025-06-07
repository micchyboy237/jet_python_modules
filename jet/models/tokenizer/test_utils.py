from jet.models.tokenizer.utils import calculate_batch_size
import psutil


class TestCalculateBatchSize:
    def test_calculate_batch_size_single_text(self):
        input_text = "This is a test sentence."
        expected = min(max(1, int((psutil.virtual_memory().available /
                       (1024 * 1024)) * 0.5 / (len(input_text) * 0.001))), 128)
        result = calculate_batch_size(input_text)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_calculate_batch_size_list_texts(self):
        input_texts = ["This is a test.", "Another sentence."]
        avg_length = sum(len(t) for t in input_texts) / len(input_texts)
        expected = min(max(1, int((psutil.virtual_memory().available /
                       (1024 * 1024)) * 0.5 / (avg_length * 0.001))), 128)
        result = calculate_batch_size(input_texts)
        assert result == expected, f"Expected {expected}, got {result}"

    def test_calculate_batch_size_with_fixed_batch(self):
        input_texts = ["This is a test.", "Another sentence."]
        fixed_batch_size = 32
        expected = 32
        result = calculate_batch_size(input_texts, fixed_batch_size)
        assert result == expected, f"Expected {expected}, got {result}"
