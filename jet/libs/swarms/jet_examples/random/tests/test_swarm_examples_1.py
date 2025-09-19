import pytest
from swarms import Agent
from typing import Callable, List, Dict, Any
from datetime import datetime
import ollama
import os
from jet.libs.swarms.jet_examples.random.swarm_examples_1 import (
    basic_agent_example,
    agent_with_tools_example,
    interactive_streaming_example,
    memory_and_docs_example,
    dynamic_temperature_example,
    artifacts_example,
    chain_of_thoughts_example,
)


class TestAgentExamples1:
    def test_basic_agent_example(self):
        # Given
        expected = "Financial report summary generated"

        # When
        result = basic_agent_example()

        # Then
        assert expected in result, f"Expected '{expected}' in result, got '{result}'"

    def test_agent_with_tools_example(self):
        # Given
        expected = "Tool processed: financial data"

        # When
        result = agent_with_tools_example()

        # Then
        assert expected in result, f"Expected '{expected}' in result, got '{result}'"

    def test_interactive_streaming_example(self):
        # Given
        expected = "story"

        # When
        result = interactive_streaming_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_memory_and_docs_example(self):
        # Given
        expected = "summary of sample_doc.txt"

        # When
        result = memory_and_docs_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_dynamic_temperature_example(self):
        # Given
        expected = "positive financial outlook"

        # When
        result = dynamic_temperature_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_artifacts_example(self):
        # Given
        expected = "markdown report"

        # When
        result = artifacts_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_chain_of_thoughts_example(self):
        # Given
        expected = "complete"

        # When
        result = chain_of_thoughts_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"
