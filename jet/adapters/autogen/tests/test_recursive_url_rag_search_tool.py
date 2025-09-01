import pytest
from unittest.mock import AsyncMock, MagicMock
from jet.adapters.autogen.recursive_url_rag_search_tool import (
    RecursiveUrlRagSearchTool,
    ToolResult,
    TextResultContent
)


@pytest.fixture
def mock_workbench():
    workbench = MagicMock()
    workbench.call_tool = AsyncMock()
    return workbench


@pytest.fixture
def tool(mock_workbench):
    return RecursiveUrlRagSearchTool(mock_workbench)


@pytest.mark.asyncio
async def test_recursive_url_rag_search_success(tool, mock_workbench):
    # Given: A query and URL with mock fetch results
    arguments = {
        "query": "python programming",
        "start_url": "https://example.com",
        "max_depth": 1,
        "max_urls": 2
    }
    expected_content = [
        {"url": "https://example.com",
            "content": "Python programming is great!", "score": 2},
        {"url": "https://example.com/subpage",
            "content": "Learn Python here.", "score": 1}
    ]
    expected_result = (
        "URL: https://example.com\nRelevance Score: 2\nContent: Python programming is great!\n\n"
        "URL: https://example.com/subpage\nRelevance Score: 1\nContent: Learn Python here."
    )

    # Mock fetch tool responses
    mock_workbench.call_tool.side_effect = [
        ToolResult(name="fetch", result=[TextResultContent(
            content="Python programming is great! <a href='/subpage'>Link</a>")], is_error=False),
        ToolResult(name="fetch", result=[TextResultContent(
            content="Learn Python here.")], is_error=False)
    ]

    # When: Running the tool
    result = await tool.run(arguments)

    # Then: Verify the result
    assert not result.is_error
    assert len(result.result) == 1
    assert result.result[0].content == expected_result


@pytest.mark.asyncio
async def test_recursive_url_rag_search_no_results(tool, mock_workbench):
    # Given: A query with no relevant content
    arguments = {
        "query": "java programming",
        "start_url": "https://example.com",
        "max_depth": 1,
        "max_urls": 1
    }
    expected_result = "No relevant content found."

    # Mock fetch tool response with irrelevant content
    mock_workbench.call_tool.return_value = ToolResult(
        name="fetch",
        result=[TextResultContent(content="This is about Ruby.")],
        is_error=False
    )

    # When: Running the tool
    result = await tool.run(arguments)

    # Then: Verify the result
    assert not result.is_error
    assert len(result.result) == 1
    assert result.result[0].content == expected_result


@pytest.mark.asyncio
async def test_recursive_url_rag_search_fetch_error(tool, mock_workbench):
    # Given: A query with a failing fetch tool
    arguments = {
        "query": "python programming",
        "start_url": "https://example.com",
        "max_depth": 1,
        "max_urls": 1
    }
    expected_result = "No relevant content found."

    # Mock fetch tool to return an error
    mock_workbench.call_tool.return_value = ToolResult(
        name="fetch",
        result=[TextResultContent(content="Error fetching page")],
        is_error=True
    )

    # When: Running the tool
    result = await tool.run(arguments)

    # Then: Verify the result
    assert not result.is_error  # Tool itself doesn't fail, just returns no results
    assert len(result.result) == 1
    assert result.result[0].content == expected_result
