import pytest
from unittest.mock import AsyncMock, patch
from jet.search.playwright import PlaywrightSearchAPIWrapper, PlaywrightExtract
from jet.search.searxng import SearchResult

@pytest.fixture
def wrapper():
    """Fixture to initialize PlaywrightSearchAPIWrapper."""
    wrapper = PlaywrightSearchAPIWrapper(
        searxng_url="http://localhost:3000",
        max_results=5,
        max_content_length=500,
        sentence_transformer_model="all-MiniLM-L6-v2"
    )
    return wrapper

@pytest.fixture(autouse=True)
def cleanup():
    """Ensure clean state before and after tests."""
    yield  # Run test
    # No specific cleanup needed, but fixture ensures scope isolation

@pytest.mark.asyncio
async def test_score_chunk_relevant_text(wrapper: PlaywrightSearchAPIWrapper):
    # Given: A relevant chunk and query
    query = "AI advancements 2025"
    chunk = "AI models advance significantly in 2025 with new algorithms."
    expected_min_score = 0.7  # Approximate threshold for high similarity

    # When: Scoring the chunk
    score = wrapper._score_chunk(chunk, query)

    # Then: Expect a high similarity score
    assert score >= expected_min_score, f"Expected score >= {expected_min_score}, got {score}"

@pytest.mark.asyncio
async def test_score_chunk_irrelevant_text(wrapper: PlaywrightSearchAPIWrapper):
    # Given: An irrelevant chunk and query
    query = "AI advancements 2025"
    chunk = "The weather in 2025 is unpredictable."
    expected_max_score = 0.3  # Approximate threshold for low similarity

    # When: Scoring the chunk
    score = wrapper._score_chunk(chunk, query)

    # Then: Expect a low similarity score
    assert score <= expected_max_score, f"Expected score <= {expected_max_score}, got {score}"

@pytest.mark.asyncio
async def test_extract_relevant_content_selects_best_chunk(wrapper: PlaywrightSearchAPIWrapper):
    # Given: A raw content string with mixed relevance and a query
    raw_content = (
        "Unrelated boilerplate about weather.\n"
        "AI models advance in 2025 with breakthroughs in neural networks.\n"
        "General news about politics."
    )
    query = "AI advancements 2025"
    max_length = 100
    expected = "AI models advance in 2025 with breakthroughs in neural networks."

    # When: Extracting relevant content
    result = wrapper._extract_relevant_content(raw_content, query, max_length)

    # Then: Expect the most relevant chunk within max_length
    assert result == expected, f"Expected '{expected}', got '{result}'"
    assert len(result) <= max_length, f"Content length {len(result)} exceeds {max_length}"

@pytest.mark.asyncio
async def test_extract_relevant_content_handles_empty_content(wrapper: PlaywrightSearchAPIWrapper):
    # Given: An empty raw content string and a query
    raw_content = ""
    query = "AI advancements 2025"
    max_length = 100
    expected = ""

    # When: Extracting relevant content
    result = wrapper._extract_relevant_content(raw_content, query, max_length)

    # Then: Expect empty content
    assert result == expected, f"Expected '{expected}', got '{result}'"

@pytest.mark.asyncio
async def test_raw_results_async_with_mocked_extract(wrapper: PlaywrightSearchAPIWrapper):
    # Given: Mocked search results and PlaywrightExtract output
    query = "AI advancements 2025"
    mock_search_results = [
        SearchResult(
            url="https://example.com",
            title="AI News 2025",
            content="Basic AI summary.",
            score=0.9
        )
    ]
    mock_extract_results = {
        "results": [{
            "url": "https://example.com",
            "raw_content": (
                "Unrelated content about sports.\n"
                "AI models in 2025 achieve breakthroughs in generative tasks.\n"
                "Footer text."
            ),
            "images": [],
            "favicon": None
        }]
    }
    expected_content = "AI models in 2025 achieve breakthroughs in generative tasks."
    expected_result = {
        "query": query,
        "follow_up_questions": None,
        "answer": None,
        "images": [],
        "results": [{
            "url": "https://example.com",
            "title": "AI News 2025",
            "content": expected_content,
            "score": 0.9,
            "raw_content": mock_extract_results["results"][0]["raw_content"],
            "images": [],
            "favicon": None
        }],
        "response_time": pytest.approx(0.0, abs=1.0)
    }

    # When: Running raw_results_async with mocked dependencies
    with patch("jet.search.searxng.search_searxng", return_value=mock_search_results):
        with patch.object(PlaywrightExtract, "_arun", AsyncMock(return_value=mock_extract_results)):
            result = await wrapper.raw_results_async(
                query=query,
                include_domains=None,
                exclude_domains=None,
                search_depth="advanced",
                include_images=False,
                time_range=None,
                topic="general",
                include_favicon=False,
                start_date=None,
                end_date=None,
                include_answer=False,
                include_raw_content=True,
                include_image_descriptions=False,
                auto_parameters=False,
                country=None
            )

    # Then: Expect relevant content extracted and correct result structure
    assert result["results"][0]["content"] == expected_content, (
        f"Expected content '{expected_content}', got '{result['results'][0]['content']}'"
    )
    assert len(result["results"][0]["content"]) <= wrapper.max_content_length, (
        f"Content length {len(result['results'][0]['content'])} exceeds {wrapper.max_content_length}"
    )
    assert result.keys() == expected_result.keys(), (
        f"Expected keys {expected_result.keys()}, got {result.keys()}"
    )
