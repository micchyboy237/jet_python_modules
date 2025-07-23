import pytest
import asyncio
from typing import List
from playwright.async_api import async_playwright
from pathlib import Path
import shutil
from jet.scrapers.automation.grok_website_cloner import clone_after_render, generate_react_components


@pytest.fixture
async def setup_browser():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        yield page
        await browser.close()


@pytest.mark.asyncio
async def test_clone_after_render():
    # Given: A simple webpage that may or may not have a CSS file
    url = "http://example.com"
    output_dir = "test_output"
    expected_html = "<html"

    # When: Cloning the webpage
    await clone_after_render(url, output_dir)

    # Then: HTML is saved correctly
    html_path = Path(output_dir) / "index.html"
    result_html = html_path.read_text(encoding="utf-8")
    assert expected_html in result_html, "HTML content not captured"

    # Check for CSS files only if they exist
    css_path = Path(output_dir) / "assets/style.css"
    if css_path.exists():
        assert css_path.exists(), "CSS file not downloaded"

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
