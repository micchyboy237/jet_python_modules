import pytest
from unittest.mock import Mock
from jet.libs.swarms.jet_examples.random.swarm_examples_custom_llm import (
    custom_ollama_agent_example,
    custom_ollama_tools_example,
    custom_ollama_streaming_example,
)

# Tests


class TestCustomOllamaAgent:
    def test_custom_ollama_agent_example(self):
        # Given
        expected = "market summary"

        # When
        result = custom_ollama_agent_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_custom_ollama_tools_example(self):
        # Given
        expected = "processed: financial data"

        # When
        result = custom_ollama_tools_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_custom_ollama_streaming_example(self):
        # Given
        expected = "financial story"

        # When
        result = custom_ollama_streaming_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
