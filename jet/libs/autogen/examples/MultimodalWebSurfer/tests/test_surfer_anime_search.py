import pytest
from unittest.mock import AsyncMock
from jet.logger import CustomLogger
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import Image
from autogen_agentchat.base._task import TaskResult
from jet.libs.autogen.examples.MultimodalWebSurfer.surfer_anime_search import search_anime_streaming_links
import PIL.Image


@pytest.mark.asyncio
async def test_search_anime_streaming_links_success(logger: CustomLogger, monkeypatch):
    """
    Test successful search for anime streaming links.
    Given: A valid anime title and season
    When: The search_anime_streaming_links function is called
    Then: It returns a list of streaming links
    """
    # Given
    anime_title = "Solo Leveling"
    season = "Season 2"
    expected_links = [
        {"name": "Crunchyroll", "url": "https://www.crunchyroll.com/solo-leveling"},
        {"name": "HiAnime", "url": "https://hianime.to/solo-leveling-season-2"}
    ]
    expected_result = "\n".join(
        [f"{link['name']} - {link['url']}" for link in expected_links])

    # Mock the MultimodalWebSurfer run method
    async def mock_run(*args, **kwargs):
        chat_message = TextMessage(content=expected_result, source="assistant")
        response = TaskResult(messages=[chat_message])
        return response

    monkeypatch.setattr(
        "jet.libs.autogen.multimodal_web_surfer.MultimodalWebSurfer.run",
        AsyncMock(side_effect=mock_run)
    )

    # When
    result = await search_anime_streaming_links(anime_title, season)

    # Then
    logger.debug(f"Result links: {result}")
    assert len(result) == len(expected_links)
    assert result == expected_links
    logger.info("Test search_anime_streaming_links_success passed.")


@pytest.mark.asyncio
async def test_search_anime_streaming_links_empty_result(logger: CustomLogger, monkeypatch):
    """
    Test search with no results returned.
    Given: A valid anime title and season
    When: The search_anime_streaming_links function is called but no links are found
    Then: It returns an empty list
    """
    # Given
    anime_title = "Solo Leveling"
    season = "Season 2"
    expected_links = []

    # Mock the MultimodalWebSurfer run method
    async def mock_run(*args, **kwargs):
        chat_message = TextMessage(
            content="No streaming links found.", source="assistant")
        response = TaskResult(messages=[chat_message])
        return response

    monkeypatch.setattr(
        "jet.libs.autogen.multimodal_web_surfer.MultimodalWebSurfer.run",
        AsyncMock(side_effect=mock_run)
    )

    # When
    result = await search_anime_streaming_links(anime_title, season)

    # Then
    logger.debug(f"Result links: {result}")
    assert len(result) == 0
    assert result == expected_links
    logger.info("Test search_anime_streaming_links_empty_result passed.")


@pytest.mark.asyncio
async def test_search_anime_streaming_links_error(logger: CustomLogger, monkeypatch):
    """
    Test search with an error during execution.
    Given: A valid anime title and season
    When: The search_anime_streaming_links function encounters an error
    Then: It returns an empty list
    """
    # Given
    anime_title = "Solo Leveling"
    season = "Season 2"
    expected_links = []

    # Mock the MultimodalWebSurfer run method to raise an error
    async def mock_run(*args, **kwargs):
        raise RuntimeError("Web search failed")

    monkeypatch.setattr(
        "jet.libs.autogen.multimodal_web_surfer.MultimodalWebSurfer.run",
        AsyncMock(side_effect=mock_run)
    )

    # When
    result = await search_anime_streaming_links(anime_title, season)

    # Then
    logger.debug(f"Result links: {result}")
    assert len(result) == 0
    assert result == expected_links
    logger.info("Test search_anime_streaming_links_error passed.")


@pytest.mark.asyncio
async def test_search_anime_streaming_links_multimodal_success(logger: CustomLogger, monkeypatch):
    """
    Test successful search with a MultiModalMessage response.
    Given: A valid anime title and season
    When: The search_anime_streaming_links function receives a TaskResult with a MultiModalMessage
    Then: It returns a list of streaming links extracted from the message content
    """
    # Given
    anime_title = "Solo Leveling"
    season = "Season 2"
    expected_links = [
        {"name": "Crunchyroll", "url": "https://www.crunchyroll.com/solo-leveling"},
        {"name": "HiAnime", "url": "https://hianime.to/solo-leveling-season-2"}
    ]
    expected_content = "\n".join(
        [f"{link['name']} - {link['url']}" for link in expected_links])

    # Mock the MultimodalWebSurfer run method
    async def mock_run(*args, **kwargs):
        chat_message = MultiModalMessage(
            content=[expected_content, Image.from_pil(
                PIL.Image.new("RGB", (100, 100)))],
            source="assistant"
        )
        response = TaskResult(messages=[chat_message])
        return response

    monkeypatch.setattr(
        "jet.libs.autogen.multimodal_web_surfer.MultimodalWebSurfer.run",
        AsyncMock(side_effect=mock_run)
    )

    # When
    result = await search_anime_streaming_links(anime_title, season)

    # Then
    logger.debug(f"Result links: {result}")
    assert len(result) == len(expected_links)
    assert result == expected_links
    logger.info("Test search_anime_streaming_links_multimodal_success passed.")
