import asyncio
import shutil
from typing import List
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest
from fake_useragent import UserAgent
from playwright.async_api import async_playwright
from jet.scrapers.automation.webpage_cloner.clone_webpage import clone_after_render, Resource


@pytest.mark.asyncio
async def test_clone_webpage_saves_html_content():
    """
    Test that clone_after_render saves the expected HTML content to the output directory.
    Given a valid URL and output directory
    When clone_after_render is called
    Then the HTML content is saved to index.html with expected content
    """
    # Given
    url = "http://example.com"
    output_dir = "test_output"
    expected_html_start = "<html"

    # When
    await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=3)

    # Then
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert expected_html_start in result_html, "Expected HTML content not found in saved file"

    # Cleanup
    shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_webpage_applies_random_user_agent():
    """
    Test that clone_after_render applies a random user agent during page navigation.
    Given a valid URL and output directory with random user agent type
    When clone_after_render is called
    Then the page uses the expected random user agent and saves HTML content
    """
    # Given
    url = "http://example.com"
    output_dir = "test_output"
    expected_html_start = "<html"
    ua = UserAgent()
    expected_user_agent = ua.random

    # When
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=expected_user_agent)
        page = await context.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        result_user_agent = await page.evaluate("() => navigator.userAgent")
        await context.close()
        await browser.close()

    await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=3)

    # Then
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert result_user_agent == expected_user_agent, "Random user agent not applied correctly"
    assert expected_html_start in result_html, "Expected HTML content not found in saved file"

    # Cleanup
    shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_webpage_handles_specific_user_agent_types():
    """
    Test that clone_after_render applies specific user agent types correctly.
    Given a valid URL and output directory with specific user agent types (web, mobile)
    When clone_after_render is called for each type
    Then the correct user agent is applied and HTML content is saved
    """
    # Given
    url = "http://example.com"
    output_dir = "test_output"
    expected_html_start = "<html"
    user_agent_types = ["web", "mobile"]
    ua = UserAgent()

    for user_agent_type in user_agent_types:
        # When
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            expected_user_agent = ua.chrome if user_agent_type == "web" else ua.random
            if user_agent_type == "mobile":
                while "Mobile" not in expected_user_agent and "Android" not in expected_user_agent and "iPhone" not in expected_user_agent:
                    expected_user_agent = ua.random
            context = await browser.new_context(user_agent=expected_user_agent)
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded")
            result_user_agent = await page.evaluate("() => navigator.userAgent")
            await context.close()
            await browser.close()

        await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type=user_agent_type, max_retries=3)

        # Then
        html_path = Path(output_dir) / "index.html"
        result_html = html_path.read_text(encoding="utf-8")
        assert result_user_agent == expected_user_agent, f"{user_agent_type} user agent not applied correctly"
        if user_agent_type == "mobile":
            assert any(keyword in result_user_agent for keyword in [
                       "Mobile", "Android", "iPhone"]), "Mobile user agent not detected"
        else:
            assert all(keyword not in result_user_agent for keyword in [
                       "Mobile", "Android", "iPhone"]), "Web user agent contains mobile keywords"
        assert expected_html_start in result_html, "Expected HTML content not found in saved file"

        # Cleanup
        shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_webpage_retries_on_navigation_failure():
    """
    Test that clone_after_render retries navigation on failure with appropriate delays.
    Given a URL with a simulated navigation failure on the first attempt
    When clone_after_render is called with retries
    Then it retries the correct number of times with unique user agents and appropriate delays
    """
    # Given
    url = "http://example.com"
    output_dir = "test_output"
    expected_html_start = "<html"
    max_retries = 2
    ua = UserAgent()
    used_user_agents: List[str] = []
    delays: List[float] = []

    async def mock_goto(self, *args, **kwargs):
        used_user_agents.append(await self.evaluate("() => navigator.userAgent"))
        if len(used_user_agents) == 1:
            raise Exception("Simulated navigation failure")
        return await self._goto(*args, **kwargs)

    original_sleep = asyncio.sleep

    async def mock_sleep(seconds: float) -> None:
        delays.append(seconds)
        await original_sleep(seconds)

    # When
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=ua.random)
        page = await context.new_page()
        page._goto = page.goto
        page.goto = mock_goto
        with patch("asyncio.sleep", side_effect=mock_sleep):
            await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=max_retries)
        await context.close()
        await browser.close()

    # Then
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert len(used_user_agents) == 2, "Expected exactly 2 retry attempts"
    assert len(set(used_user_agents)
               ) == 2, "User agents were not unique across retries"
    assert len(delays) == 1, "Expected exactly 1 delay"
    assert 1 <= delays[0] <= 5, f"Delay {delays[0]} not in expected range (1-5 seconds)"
    assert expected_html_start in result_html, "Expected HTML content not found in saved file"

    # Cleanup
    shutil.rmtree(output_dir)
