from typing import List, Union, Literal
from pydantic.json_schema import JsonSchemaValue
import pytest
from jet.llm.mlx.mlx_types import Message
from jet.llm.mlx.mlx_utils import process_response_format


class TestProcessResponseFormat:
    """Test suite for process_response_format function."""

    def test_string_input_text_format(self):
        """Test string input with text response format returns unchanged."""
        # Given
        input_prompt: str = "What is the capital of France?"
        response_format: Literal["text"] = "text"
        expected_result: str = input_prompt

        # When
        result = process_response_format(input_prompt, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_string_input_json_format(self):
        """Test string input with json response format adds JSON instruction."""
        # Given
        input_prompt: str = "What is the capital of France?"
        response_format: Literal["json"] = "json"
        expected_instruction: str = (
            "Return the response as a JSON object containing only the data fields defined in the schema, "
            "without including the schema itself or any additional metadata"
        )
        expected_result: str = f"{expected_instruction}\n\n{input_prompt}"

        # When
        result = process_response_format(input_prompt, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_string_input_json_schema(self):
        """Test string input with JSON schema adds schema and example."""
        # Given
        input_prompt: str = "Provide user details."
        response_format: JsonSchemaValue = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        expected_instruction: str = (
            'Return the response as a JSON object containing only the data fields defined in the schema, '
            'without including the schema itself or any additional metadata\n'
            '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}\n'
            'For example, return only {"name": "value", "age": 0}.'
        )
        expected_result: str = f"{expected_instruction}\n\n{input_prompt}"

        # When
        result = process_response_format(input_prompt, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_message_list_text_format(self):
        """Test message list with text response format returns unchanged."""
        # Given
        input_messages: List[Message] = [
            {"role": "user", "content": "Hello, how are you?", "tool_calls": None}
        ]
        response_format: Literal["text"] = "text"
        expected_result: List[Message] = input_messages.copy()

        # When
        result = process_response_format(input_messages, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_message_list_json_format_no_system(self):
        """Test message list with json format adds system message."""
        # Given
        input_messages: List[Message] = [
            {"role": "user", "content": "Hello, how are you?", "tool_calls": None}
        ]
        response_format: Literal["json"] = "json"
        expected_instruction: str = (
            "Return the response as a JSON object containing only the data fields defined in the schema, "
            "without including the schema itself or any additional metadata"
        )
        expected_result: List[Message] = [
            {"role": "system", "content": expected_instruction, "tool_calls": None},
            {"role": "user", "content": "Hello, how are you?", "tool_calls": None}
        ]

        # When
        result = process_response_format(input_messages, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_message_list_json_schema_no_system(self):
        """Test message list with JSON schema adds system message with schema and example."""
        # Given
        input_messages: List[Message] = [
            {"role": "user", "content": "Provide user details.", "tool_calls": None}
        ]
        response_format: JsonSchemaValue = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        expected_instruction: str = (
            'Return the response as a JSON object containing only the data fields defined in the schema, '
            'without including the schema itself or any additional metadata\n'
            '{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}\n'
            'For example, return only {"name": "value", "age": 0}.'
        )
        expected_result: List[Message] = [
            {"role": "system", "content": expected_instruction, "tool_calls": None},
            {"role": "user", "content": "Provide user details.", "tool_calls": None}
        ]

        # When
        result = process_response_format(input_messages, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"

    def test_message_list_json_format_with_system(self):
        """Test message list with existing system message appends JSON instruction."""
        # Given
        input_messages: List[Message] = [
            {"role": "system", "content": "You are a helpful assistant.", "tool_calls": None},
            {"role": "user", "content": "Hello, how are you?", "tool_calls": None}
        ]
        response_format: Literal["json"] = "json"
        expected_instruction: str = (
            "Return the response as a JSON object containing only the data fields defined in the schema, "
            "without including the schema itself or any additional metadata"
        )
        expected_result: List[Message] = [
            {
                "role": "system",
                "content": f"You are a helpful assistant.\n\n{expected_instruction}",
                "tool_calls": None
            },
            {"role": "user", "content": "Hello, how are you?", "tool_calls": None}
        ]

        # When
        result = process_response_format(input_messages, response_format)

        # Then
        assert result == expected_result, f"Expected {expected_result}, got {result}"
