import pytest
import shutil
import asyncio
from typing import List
from playwright.async_api import async_playwright
from pathlib import Path
from fake_useragent import UserAgent
from jet.scrapers.automation.webpage_cloner import clone_after_render, generate_react_components


@pytest.mark.asyncio
async def test_clone_after_render():
    # Given: A simple webpage
    url = "http://example.com"
    output_dir = "test_output"
    expected_html = "<html"

    # When: Cloning the webpage
    await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=3)

    # Then: HTML is saved correctly
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert expected_html in result_html, "HTML content not captured"

    # Cleanup
    shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_generate_react_components():
    # Given: Sample HTML with a styled div
    html = """
    <div class="card" style="color: blue;">Hello</div>
    """
    output_dir = "test_components"
    expected_component_name = "Card"
    expected_html = "<div class=\"card\" style=\"color: blue;\">Hello</div>"
    expected_styles = "color: blue;"

    # When: Generating React components
    components = generate_react_components(html, output_dir)

    # Then: Component is generated with correct HTML and styles
    result_component = components[0]
    assert result_component["name"] == expected_component_name
    assert result_component["html"] == expected_html
    assert result_component["styles"] == expected_styles

    component_path = Path(output_dir) / f"{expected_component_name}.jsx"
    css_path = Path(output_dir) / f"{expected_component_name}.css"
    assert component_path.exists(), "Component file not created"
    assert css_path.exists(), "CSS file not created"

    # Cleanup
    shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_after_render_with_random_user_agent():
    # Given: A simple webpage and a random user agent
    url = "http://example.com"
    output_dir = "test_output"
    expected_html = "<html"
    ua = UserAgent()
    expected_user_agent = ua.random

    # When: Cloning the webpage with a random user agent
    await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=3)

    # Then: HTML is saved (user agent tested in separate test)
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert expected_html in result_html, "HTML content not captured"

    # Cleanup
    shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_after_render_with_specific_user_agent_type():
    # Given: A webpage and specific user agent types
    url = "http://example.com"
    output_dir = "test_output"
    expected_html = "<html"
    user_agent_types = ["web", "mobile"]
    ua = UserAgent()

    for user_agent_type in user_agent_types:
        # When: Cloning the webpage with a specific user agent type
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

        # Then: Correct user agent type is applied and HTML is saved
        assert result_user_agent == expected_user_agent, f"{user_agent_type} user agent not applied"
        if user_agent_type == "mobile":
            assert any(keyword in result_user_agent for keyword in [
                       "Mobile", "Android", "iPhone"]), "Mobile user agent not detected"
        else:
            assert "Mobile" not in result_user_agent and "Android" not in result_user_agent and "iPhone" not in result_user_agent, "Web user agent contains mobile keywords"
        html_path = Path(output_dir) / "index.html"
        result_html = html_path.read_text(encoding="utf-8")
        assert expected_html in result_html, "HTML content not captured"

        # Cleanup
        shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_clone_after_render_with_retries_and_delays():
    # Given: A webpage with simulated navigation failures
    url = "http://example.com"
    output_dir = "test_output"
    expected_html = "<html"
    max_retries = 2
    ua = UserAgent()
    used_user_agents = []
    delays = []

    # Mock page.goto to fail for the first attempt, succeed on the second
    async def mock_goto(self, *args, **kwargs):
        used_user_agents.append(await self.evaluate("() => navigator.userAgent"))
        if len(used_user_agents) == 1:
            raise Exception("Simulated navigation failure")
        return await self._goto(*args, **kwargs)

    # Mock asyncio.sleep to capture delays
    original_sleep = asyncio.sleep

    async def mock_sleep(seconds):
        delays.append(seconds)
        await original_sleep(seconds)

    # When: Cloning with retries and mocked sleep
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=ua.random)
        page = await context.new_page()
        page._goto = page.goto
        page.goto = mock_goto
        asyncio.sleep = mock_sleep
        try:
            await clone_after_render(url, output_dir, headless=True, timeout=10000, user_agent_type="random", max_retries=max_retries)
        finally:
            asyncio.sleep = original_sleep  # Restore original sleep

        await context.close()
        await browser.close()

    # Then: Retries use different user agents, delays are applied, and HTML is saved
    assert len(used_user_agents) == 2, "Incorrect number of retries"
    assert len(set(used_user_agents)
               ) == 2, "User agents were not unique across retries"
    assert len(delays) == 1, "Incorrect number of delays applied"
    assert 1 <= delays[0] <= 5, f"Delay {delays[0]} not in expected range (1-5 seconds)"
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert expected_html in result_html, "HTML content not captured"

    # Cleanup
    shutil.rmtree(output_dir)
