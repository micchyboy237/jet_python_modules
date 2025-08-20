# jet_server/playwright_mcp/server/tests/test_parse_tool_requests.py
import pytest
from typing import List
from jet.logger import CustomLogger
from jet.servers.mcp.server.mcp_agent import ToolRequest, parse_tool_requests


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
        """
        # Given: A response with a single valid JSON tool request
        response = '{"tool": "navigate_to_url", "arguments": {"url": "https://example.com"}}'
        expected_requests = [
            ToolRequest(tool="navigate_to_url", arguments={
                        "url": "https://example.com"})
        ]

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: The parsed request should match the expected ToolRequest
        assert len(result) == len(expected_requests)
        assert result[0].tool == expected_requests[0].tool
        assert result[0].arguments == expected_requests[0].arguments

        # Then: The logger should record one JSON object found
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 1 JSON objects in response" in log_content
        assert "Valid tool request: tool='navigate_to_url'" in log_content

    def test_parse_multiple_valid_tool_requests(self, mock_logger):
        """
        Test parsing multiple valid JSON tool requests across multiple lines.
        """
        # Given: A response with multiple valid JSON tool requests
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

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: All parsed requests should match the expected ToolRequests
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments

        # Then: The logger should record two JSON objects found
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 2 JSON objects in response" in log_content
        assert "Valid tool request: tool='navigate_to_url'" in log_content
        assert "Valid tool request: tool='summarize_text'" in log_content

    def test_parse_mixed_valid_and_invalid_json(self, mock_logger):
        """
        Test parsing a response with both valid and invalid JSON objects.
        """
        # Given: A response with valid and invalid JSON objects
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

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: Only valid requests should be returned
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments

        # Then: The logger should record three JSON objects and one invalid
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 3 JSON objects in response" in log_content
        assert "Invalid JSON object skipped" in log_content
        assert "Valid tool request: tool='navigate_to_url'" in log_content
        assert "Valid tool request: tool='summarize_text'" in log_content

    def test_parse_empty_response(self, mock_logger):
        """
        Test parsing an empty response or one with no JSON objects.
        """
        # Given: An empty response
        response = ""
        expected_requests = []

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: No requests should be returned
        assert result == expected_requests

        # Then: The logger should record zero JSON objects found
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 0 JSON objects in response" in log_content

    def test_parse_invalid_tool_request_schema(self, mock_logger):
        """
        Test parsing a JSON object that doesn't match the ToolRequest schema.
        """
        # Given: A response with a JSON object missing required fields
        response = '{"invalid_field": "value"}'
        expected_requests = []

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: No valid requests should be returned
        assert result == expected_requests

        # Then: The logger should record the schema validation error
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 1 JSON objects in response" in log_content
        assert "Invalid JSON object skipped" in log_content
        assert "validation errors for ToolRequest" in log_content

    def test_parse_json_single_and_multiline(self, mock_logger):
        """
        Test parsing one JSON object across multiple lines and one on a single line.
        """
        # Given: A response with one JSON object across multiple lines and one on a single line
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

        # When: Parsing the response
        result = parse_tool_requests(response, mock_logger)

        # Then: All parsed requests should match the expected ToolRequests
        assert len(result) == len(expected_requests)
        for parsed, expected in zip(result, expected_requests):
            assert parsed.tool == expected.tool
            assert parsed.arguments == expected.arguments

        # Then: The logger should record two JSON objects found
        with open(mock_logger.log_file, "r") as f:
            log_content = f.read()
        assert "Found 2 JSON objects in response" in log_content
        assert "Valid tool request: tool='navigate_to_url'" in log_content
        assert "Valid tool request: tool='summarize_text'" in log_content
