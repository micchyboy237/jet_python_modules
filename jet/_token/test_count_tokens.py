from llama_index.core.base.llms.types import ChatMessage
from jet._token.token_utils import token_counter
from jet.llm.llm_types import Message

class TestTokenCounter:
    def test_count_tokens_single_string(self):
        # Given
        input_text = "Hello, this is a test."
        model = "llama3.2"
        expected_count = 8  # Approximate token count for the input text

        # When
        result = token_counter(input_text, model)

        # Then
        assert isinstance(result, int), "Result should be an integer for single string input"
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"

    def test_count_tokens_empty_input(self):
        # Given
        input_text = ""
        model = "llama3.2"
        expected_count = 0

        # When
        result = token_counter(input_text, model)

        # Then
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"

    def test_count_tokens_list_of_strings(self):
        # Given
        input_texts = ["Hello world", "This is a test"]
        model = "llama3.2"
        expected_counts = [3, 5]  # Approximate token counts for each string
        expected_total = sum(expected_counts)

        # When
        result = token_counter(input_texts, model, prevent_total=False)

        # Then
        assert isinstance(result, int), "Result should be total count when prevent_total is False"
        assert result == expected_total, f"Expected total {expected_total} tokens, got {result}"

    def test_count_tokens_list_of_strings_prevent_total(self):
        # Given
        input_texts = ["Hello world", "This is a test"]
        model = "llama3.2"
        expected_counts = [3, 5]  # Approximate token counts for each string

        # When
        result = token_counter(input_texts, model, prevent_total=True)

        # Then
        assert isinstance(result, list), "Result should be a list when prevent_total is True"
        assert len(result) == len(input_texts), f"Expected list of length {len(input_texts)}, got {len(result)}"
        assert result == expected_counts, f"Expected {expected_counts}, got {result}"

    def test_count_tokens_chat_message(self):
        # Given
        input_message = [ChatMessage(content="Hello, how are you?")]
        model = "llama3.2"
        expected_count = 7  # Approximate token count for the message content

        # When
        result = token_counter(input_message, model)

        # Then
        assert isinstance(result, int), "Result should be an integer for ChatMessage input"
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"

    def test_count_tokens_message_dict(self):
        # Given
        input_message = [Message(content="This is a message", role="user")]
        model = "llama3.2"
        expected_count = 5  # Approximate token count for the message content

        # When
        result = token_counter(input_message, model)

        # Then
        assert isinstance(result, int), "Result should be an integer for Message dict input"
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"

    def test_count_tokens_dict(self):
        # Given
        input_dict = {"content": "This is a test dictionary"}
        model = "llama3.2"
        expected_count = 6  # Approximate token count for the dictionary content

        # When
        result = token_counter(input_dict, model)

        # Then
        assert isinstance(result, int), "Result should be an integer for dict input"
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"

    def test_count_tokens_list_of_mixed_types(self):
        # Given
        input_mixed = [
            "Hello world",
            ChatMessage(content="How are you?"),
            Message(content="Test message", role="user"),
            {"content": "Dictionary content"}
        ]
        model = "llama3.2"
        expected_counts = [3, 4, 3, 4]  # Approximate token counts for each item
        expected_total = sum(expected_counts)

        # When
        result = token_counter(input_mixed, model, prevent_total=False)

        # Then
        assert isinstance(result, int), "Result should be total count for mixed input"
        assert result == expected_total, f"Expected total {expected_total} tokens, got {result}"

    def test_count_tokens_list_of_mixed_types_prevent_total(self):
        # Given
        input_mixed = [ 
            "Hello world",
            ChatMessage(content="How are you?"),
            Message(content="Test message", role="user"),
            {"content": "Dictionary content"}
        ]
        model = "llama3.2"
        expected_counts = [3, 5, 3, 3]  # Approximate token counts for each item

        # When
        result = token_counter(input_mixed, model, prevent_total=True)

        # Then
        assert isinstance(result, list), "Result should be a list when prevent_total is True"
        assert len(result) == len(input_mixed), f"Expected list of length {len(input_mixed)}, got {len(result)}"
        assert result == expected_counts, f"Expected {expected_counts}, got {result}"

    def test_count_tokens_invalid_model_fallback(self):
        # Given
        input_text = "Hello world"
        model = "non_existent_model"
        expected_count = 2  # Approximate token count for "Hello world" with cl100k_base

        # When
        result = token_counter(input_text, model)

        # Then
        assert isinstance(result, int), "Result should be an integer for single string input"
        assert result == expected_count, f"Expected {expected_count} tokens, got {result}"
