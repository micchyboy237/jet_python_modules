import pytest
from swarms import Agent
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
import ollama
import os
import json
import yaml
import xml.etree.ElementTree as ET
from unittest.mock import Mock
from jet.libs.swarms.jet_examples.swarm_examples_2 import (
    template_retry_example,
    stopping_token_dynamic_loops_example,
    dashboard_exit_command_example,
    sop_autosave_example,
    self_healing_code_interpreter_example,
    multi_modal_pdf_example,
    callbacks_metadata_example,
    search_evaluator_example,
    logging_custom_loop_example,
    function_calling_cleaner_example,
    planning_tools_prompt_example,
    advanced_parameters_example,
    scheduled_workspace_example,
    history_feedback_example,
)


# Tests
class TestAgentExtendedExamples:
    def test_template_retry_example(self):
        # Given
        expected = "market trends"

        # When
        result = template_retry_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_stopping_token_dynamic_loops_example(self):
        # Given
        expected = "<stop>"

        # When
        result = stopping_token_dynamic_loops_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_dashboard_exit_command_example(self):
        # Given
        expected = "dashboard"

        # When
        result = dashboard_exit_command_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_sop_autosave_example(self):
        # Given
        expected = "financial analysis"

        # When
        result = sop_autosave_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_self_healing_code_interpreter_example(self):
        # Given
        expected = "financial data"

        # When
        result = self_healing_code_interpreter_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_multi_modal_pdf_example(self):
        # Given
        expected = "pdf content"

        # When
        result = multi_modal_pdf_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_callbacks_metadata_example(self):
        # Given
        expected = "metadata"

        # When
        result = callbacks_metadata_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_search_evaluator_example(self):
        # Given
        expected = "result for market trends"

        # When
        result = search_evaluator_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_logging_custom_loop_example(self):
        # Given
        expected = "report"

        # When
        result = logging_custom_loop_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_function_calling_cleaner_example(self):
        # Given
        expected = "JSON RESPONSE"

        # When
        result = function_calling_cleaner_example()

        # Then
        assert expected in result, f"Expected '{expected}' in result, got '{result}'"

    def test_planning_tools_prompt_example(self):
        # Given
        expected = "financial analysis"

        # When
        result = planning_tools_prompt_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_advanced_parameters_example(self):
        # Given
        expected = "financial summary"

        # When
        result = advanced_parameters_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_scheduled_workspace_example(self):
        # Given
        expected = "scheduled task"

        # When
        result = scheduled_workspace_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"

    def test_bulk_concurrent_example(self):
        # Given
        expected = ["summary", "analysis"]

        # When
        result = bulk_concurrent_example()

        # Then
        assert all(exp in str(result).lower()
                   for exp in expected), f"Expected {expected} in result, got '{result}'"

    def test_history_feedback_example(self):
        # Given
        expected = "report"

        # When
        result = history_feedback_example()

        # Then
        assert expected in result.lower(
        ), f"Expected '{expected}' in result, got '{result}'"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
