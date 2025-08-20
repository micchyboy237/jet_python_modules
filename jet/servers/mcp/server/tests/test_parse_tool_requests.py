import pytest
from typing import List
from jet.logger import CustomLogger
from jet.servers.mcp.server.mcp_classes import ToolRequest
from jet.servers.mcp.server.utils import parse_tool_requests


@pytest.fixture
def mock_logger(tmp_path):
    """
    Fixture to create a temporary logger for testing.
    """
    log_file = tmp_path / "test_parse_tool_requests.log"
    logger = CustomLogger(str(log_file), overwrite=True)
    return logger


class TestParseToolRequests:
    """
    Tests for the parse_tool_requests function, covering common JSON parsing scenarios.
    """

    def test_parse_single_valid_tool_request(self, mock_logger):
        """
        Test parsing a single valid JSON tool request on a single line.

        Given: A response string with a single valid JSON tool request
        When: parse_tool_requests is called with the response and logger
        Then: A list with one valid ToolRequest object is returned
        """
        response = '{"tool": "navigate_to_url", "arguments": {"url": "https://example.com"}}'
        expected_requests = [
            ToolRequest(tool="navigate_to_url", arguments={
                        "url": "https://example.com"})
        ]
        result = parse_tool_requests(response, mock_logger)
        assert len(result) == len(expected_requests)
        assert result[0].tool == expected_requests[0].tool
        assert result[0].arguments == expected_requests[0].arguments

    def test_parse_multiple_valid_tool_requests(self, mock_logger):
        """
        Test parsing multiple valid JSON tool requests across multiple lines.

        Given: A response string with multiple valid JSON tool requests
        When: parse_tool_requests is called with the response and logger
        Then: A list with all valid ToolRequest objects is returned
        """
        response = (
            '{"tool": "navigate_to_url", "arguments": {"url": "https://example.com"}}\n'
            '{"tool": "summarize_text", "arguments": {"text": "Sample text for summarization", "max_words": 100}}'
        )
        expected_requests = [
            ToolRequest(tool="navigate_to_url", arguments={
                        "url": "https://example.com"}),
            ToolRequest(tool="summarize_text", arguments={
                        "text": "Sample text for summarization", "max_words": 100})
        ]
        result = parse_tool_requests(response, mock_logger)
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments

    def test_parse_mixed_valid_and_invalid_json(self, mock_logger):
        """
        Test parsing a response with both valid and invalid JSON objects.

        Given: A response string with valid and invalid JSON objects
        When: parse_tool_requests is called with the response and logger
        Then: A list with only valid ToolRequest objects is returned
        """
        response = (
            '{"tool": "navigate_to_url", "arguments": {"url": "https://example.com"}}\n'
            '{invalid_json}\n'
            '{"tool": "summarize_text", "arguments": {"text": "Sample text", "max_words": 100}}'
        )
        expected_requests = [
            ToolRequest(tool="navigate_to_url", arguments={
                        "url": "https://example.com"}),
            ToolRequest(tool="summarize_text", arguments={
                        "text": "Sample text", "max_words": 100})
        ]
        result = parse_tool_requests(response, mock_logger)
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments

    def test_parse_empty_response(self, mock_logger):
        """
        Test parsing an empty response or one with no JSON objects.

        Given: An empty response string
        When: parse_tool_requests is called with the response and logger
        Then: An empty list is returned
        """
        response = ""
        expected_requests = []
        result = parse_tool_requests(response, mock_logger)
        assert result == expected_requests

    def test_parse_invalid_tool_request_schema(self, mock_logger):
        """
        Test parsing a JSON object that doesn't match the ToolRequest schema.

        Given: A response string with a JSON object missing required fields
        When: parse_tool_requests is called with the response and logger
        Then: An empty list is returned
        """
        response = '{"invalid_field": "value"}'
        expected_requests = []
        result = parse_tool_requests(response, mock_logger)
        assert result == expected_requests

    def test_parse_json_single_and_multiline(self, mock_logger):
        """
        Test parsing one JSON object across multiple lines and one on a single line.

        Given: A response string with JSON objects in single and multiline formats
        When: parse_tool_requests is called with the response and logger
        Then: A list with all valid ToolRequest objects is returned
        """
        response = (
            '{\n'
            '  "tool": "navigate_to_url",\n'
            '  "arguments": {\n'
            '    "url": "https://example.com"\n'
            '  }\n'
            '}\n'
            '{"tool": "summarize_text", "arguments": {"text": "Sample text for summarization", "max_words": 100}}'
        )
        expected_requests = [
            ToolRequest(tool="navigate_to_url", arguments={
                        "url": "https://example.com"}),
            ToolRequest(tool="summarize_text", arguments={
                        "text": "Sample text for summarization", "max_words": 100})
        ]
        result = parse_tool_requests(response, mock_logger)
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments
