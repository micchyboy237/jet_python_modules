import asyncio
import json
from jet.servers.mcp.llm_client import query_llm
from jet.servers.mcp.mcp_classes import ToolRequest


def test_chained_tool_requests(mocker):
    """Given a prompt requiring chained tools, when query_llm is called, then tools are executed with correct arguments."""
    prompt = "Navigate to https://example.com and summarize its content."
    tool_info = [
        {"name": "navigate_to_url", "schema": {"url": {"type": "string"}}, "outputSchema": {
            "title": {"type": "string"}, "text_content": {"type": "string"}}},
        {"name": "summarize_text", "schema": {"text": {"type": "string"}, "max_words": {"type": "integer"}},
            "outputSchema": {"summary": {"type": "string"}, "word_count": {"type": "integer"}}}
    ]
    mocker.patch("mcp_agent.parse_tool_requests", return_value=[
        ToolRequest(tool="navigate_to_url", arguments={
                    "url": "https://example.com"}),
        ToolRequest(tool="summarize_text", arguments={
                    "text": "{{navigate_to_url.text_content}}", "max_words": 100})
    ])
    mocker.patch("mcp_agent.execute_tool", side_effect=[
        json.dumps(
            {"title": "Example", "text_content": "This is example content."}),
        json.dumps({"summary": "Example content", "word_count": 2})
    ])
    mocker.patch("mcp_agent.generate", return_value="Final response")
    result, messages = asyncio.run(query_llm(prompt, tool_info))
    expected_messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": mocker.ANY},
        {"role": "user", "content": 'Tool result: {"title": "Example", "text_content": "This is example content."}'},
        {"role": "user", "content": 'Tool result: {"summary": "Example content", "word_count": 2}'}
    ]
    assert result == "Final response"
    assert len(messages) == len(expected_messages)
    assert messages[2]["content"] == expected_messages[2]["content"]
    assert messages[3]["content"] == expected_messages[3]["content"]
