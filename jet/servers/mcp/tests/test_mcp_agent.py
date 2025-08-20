import pytest
import asyncio
import json
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch
from jet.servers.mcp.mcp_agent import query_llm, format_tool_request_messages
from jet.servers.mcp.mcp_classes import ToolInfo, ExecutedToolResponse
from jet.models.model_types import LLMModelType
from jet.llm.mlx.mlx_types import Message


@pytest.fixture
def mock_tools():
    return [
        ToolInfo(
            name="read_file",
            description="Read the contents of a file.",
            schema={"file_path": str, "encoding": str},
            outputSchema={"text": str}
        )
    ]


@pytest.fixture
def output_dir(tmp_path):
    return str(tmp_path / "output")


@pytest.mark.asyncio
async def test_query_llm_single_tool_call(mock_tools, output_dir):
    # Given: A prompt that triggers a single tool call
    prompt = "Read file example.txt"
    model = "qwen3-1.7b-4bit"
    previous_messages: List[Message] = []
    tool_request = {"tool": "read_file", "arguments": {
        "file_path": "example.txt", "encoding": "utf-8"}}
    tool_response = ExecutedToolResponse(
        isError=False, meta={}, content={"text": "File content"})
    expected_response = "File content read successfully"
    expected_messages: List[Message] = [
        # System prompt with tool description
        {"role": "system", "content": "..."},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": json.dumps(tool_request)},
        {"role": "user", "content": f"Tool response: {tool_response}"},
    ]

    # When: query_llm is called with a tool call response
    with patch("jet.servers.mcp.mcp_agent.generate_response", return_value=json.dumps(tool_request)) as mock_generate, \
         patch("jet.servers.mcp.mcp_agent.query_tool_requests", AsyncMock(return_value=[tool_request])) as mock_query_requests, \
         patch("jet.servers.mcp.mcp_agent.query_tool_responses", AsyncMock(return_value=[{"isError": False, "meta": {}, "structuredContent": {"text": "File content"}}])) as mock_query_responses, \
         patch("jet.servers.mcp.mcp_agent.format_tool_response_messages", return_value=expected_messages) as mock_format_response, \
         patch("jet.servers.mcp.mcp_agent.save_file") as mock_save_file:
        # First tool call, then final response
        mock_generate.side_effect = [
            json.dumps(tool_request), expected_response]
        result_response, result_messages = await query_llm(
            prompt=prompt,
            model=model,
            tools=mock_tools,
            output_dir=output_dir,
            previous_messages=previous_messages
        )

    # Then: The response and messages match the expected output
    assert result_response == expected_response
    assert len(result_messages) == len(expected_messages)
    assert mock_query_requests.call_count == 2  # Called twice due to recursion
    assert mock_query_responses.call_count == 1  # Only one tool response
    # At least tool_requests and tool_responses saved
    assert mock_save_file.call_count >= 2


@pytest.mark.asyncio
async def test_query_llm_no_tool_call(mock_tools, output_dir):
    # Given: A prompt that does not require a tool call
    prompt = "
