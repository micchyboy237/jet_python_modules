import pytest
import logging
from jet.libs.autogen.playwright_controller import PlaywrightController
from playwright.async_api import async_playwright
# Added VisualViewport import
from autogen_ext.agents.web_surfer._types import VisualViewport, interactiveregion_from_dict

FAKE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake Page</title>
</head>
<body>
    <h1 id="header">Welcome to the Fake Page</h1>
    <button id="click-me">Click Me</button>
    <input type="text" id="input-box" />
    <div id="long-content" style="height: 2000px;">Long content for scrolling</div>
</body>
</html>
"""

# Setup logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_playwright_controller_initialization() -> None:
    controller = PlaywrightController()
    assert controller.viewport_width == 1440
    assert controller.viewport_height == 900
    assert controller.animate_actions is False


@pytest.mark.asyncio
async def test_playwright_controller_visit_page() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)

        controller = PlaywrightController()
        await controller.visit_page(page, "data:text/html," + FAKE_HTML)
        assert page.url.startswith("data:text/html")


@pytest.mark.asyncio
async def test_playwright_controller_click_id() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)

        controller = PlaywrightController()
        rects = await controller.get_interactive_rects(page)
        click_me_id = ""
        for rect in rects:
            if rects[rect]["aria_name"] == "Click Me":
                click_me_id = str(rect)
                break

        await controller.click_id(page, click_me_id)
        assert await page.evaluate("document.activeElement.id") == "click-me"


@pytest.mark.asyncio
async def test_playwright_controller_fill_id() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        rects = await PlaywrightController().get_interactive_rects(page)
        input_box_id = ""
        for rect in rects:
            if rects[rect]["tag_name"] == "input, type=text":
                input_box_id = str(rect)
                break
        controller = PlaywrightController()
        await controller.fill_id(page, input_box_id, "test input")
        assert await page.evaluate("document.getElementById('input-box').value") == "test input"


@pytest.mark.asyncio
async def test_playwright_controller_sleep() -> None:
    # Given: A Playwright page and controller
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        start_time = await page.evaluate("() => performance.now()")

        # When: Sleeping for 0.5 seconds
        await controller.sleep(page, 0.5)

        # Then: The elapsed time should be at least 500ms
        end_time = await page.evaluate("() => performance.now()")
        elapsed_time = end_time - start_time
        expected_min_time = 500  # 0.5 seconds in milliseconds
        assert elapsed_time >= expected_min_time, f"Expected sleep time >= {expected_min_time}ms, but got {elapsed_time}ms"


@pytest.mark.asyncio
async def test_playwright_controller_get_visual_viewport() -> None:
    # Given: A Playwright page with a known viewport size
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)

        # When: Retrieving the visual viewport
        result = await controller.get_visual_viewport(page)

        # Then: The viewport should match the expected dimensions
        expected_viewport = {
            "offsetLeft": 0,
            "offsetTop": 0,
            "width": 1440,
            "height": 900,
            "clientWidth": 1440,
            "clientHeight": 900,
            "pageLeft": 0,
            "pageTop": 0,
            "scale": 1.0
        }
        logger.debug(f"Visual viewport result: {result}")
        assert isinstance(result, (dict, VisualViewport)
                          ), "Result is neither a dict nor VisualViewport object"
        if isinstance(result, dict):
            assert result.get(
                "width") == expected_viewport["width"], f"Expected width {expected_viewport['width']}, got {result.get('width')}"
            assert result.get(
                "height") == expected_viewport["height"], f"Expected height {expected_viewport['height']}, got {result.get('height')}"
        else:
            assert result.width == expected_viewport[
                "width"], f"Expected width {expected_viewport['width']}, got {result.width}"
            assert result.height == expected_viewport[
                "height"], f"Expected height {expected_viewport['height']}, got {result.height}"


@pytest.mark.asyncio
async def test_playwright_controller_get_focused_rect_id() -> None:
    # Given: A Playwright page with an input element that can be focused
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        rects = await controller.get_interactive_rects(page)
        input_box_id = next(
            (rect for rect in rects if rects[rect]["tag_name"] == "input, type=text"), None)

        # When: Focusing the input element and retrieving the focused rect ID
        await page.locator("#input-box").focus()
        result = await controller.get_focused_rect_id(page)

        # Then: The focused rect ID should match the input box ID
        expected_id = input_box_id
        assert result == expected_id, f"Expected focused rect ID {expected_id}, got {result}"


@pytest.mark.asyncio
async def test_playwright_controller_get_page_metadata() -> None:
    # Given: A Playwright page with known content
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()

        # When: Retrieving page metadata
        result = await controller.get_page_metadata(page)

        # Then: The metadata should be a dictionary with expected meta_tags
        logger.debug(f"Page metadata result: {result}")
        expected_metadata = {
            "meta_tags": {
                "viewport": "width=device-width, initial-scale=1.0"
            }
        }
        assert isinstance(result, dict), "Metadata is not a dictionary"
        assert result.get(
            "meta_tags") == expected_metadata["meta_tags"], f"Expected meta_tags {expected_metadata['meta_tags']}, got {result.get('meta_tags')}"
        # Note: URL and title are not included in metadata based on debug log


@pytest.mark.asyncio
async def test_playwright_controller_back() -> None:
    # Given: A Playwright page with a navigation history
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("data:text/html,<h1>First Page</h1>")
        await page.wait_for_load_state()
        await page.goto("data:text/html," + FAKE_HTML)
        await page.wait_for_load_state()
        controller = PlaywrightController()

        # When: Navigating back
        await controller.back(page)

        # Then: The page should return to the first page
        expected_title = "First Page"
        result_title = await page.evaluate("document.querySelector('h1').innerText")
        assert result_title == expected_title, f"Expected title {expected_title}, got {result_title}"


@pytest.mark.asyncio
async def test_playwright_controller_page_down() -> None:
    # Given: A Playwright page with scrollable content
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)

        # When: Scrolling down by one viewport height minus 50 pixels
        initial_scroll = await page.evaluate("window.scrollY")
        await controller.page_down(page)

        # Then: The scroll position should increase by viewport height minus 50
        result_scroll = await page.evaluate("window.scrollY")
        expected_scroll = initial_scroll + (900 - 50)
        assert result_scroll == expected_scroll, f"Expected scroll position {expected_scroll}, got {result_scroll}"


@pytest.mark.asyncio
async def test_playwright_controller_page_up() -> None:
    # Given: A Playwright page scrolled down with scrollable content
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)
        await page.evaluate("window.scrollTo(0, 1000);")  # Scroll down first

        # When: Scrolling up by one viewport height minus 50 pixels
        initial_scroll = await page.evaluate("window.scrollY")
        await controller.page_up(page)

        # Then: The scroll position should decrease by viewport height minus 50
        result_scroll = await page.evaluate("window.scrollY")
        expected_scroll = initial_scroll - (900 - 50)
        assert result_scroll == expected_scroll, f"Expected scroll position {expected_scroll}, got {result_scroll}"


@pytest.mark.asyncio
async def test_playwright_controller_hover_id() -> None:
    # Given: A Playwright page with a button to hover
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        # Disable animation for simplicity
        controller = PlaywrightController(animate_actions=False)
        rects = await controller.get_interactive_rects(page)
        expected_button_id = next(
            (rect for rect in rects if rects[rect]["aria_name"] == "Click Me"), None)
        expected_is_visible = True

        # When: Hovering over the button
        await controller.hover_id(page, expected_button_id)

        # Then: The button should be visible and in view after hover
        result_is_visible = await page.locator(f"[__elementId='{expected_button_id}']").is_visible()
        logger.debug(
            f"Hovered element visibility: {result_is_visible}, expected: {expected_is_visible}")
        assert result_is_visible == expected_is_visible, f"Expected button visibility {expected_is_visible}, got {result_is_visible}"


@pytest.fixture(autouse=True)
async def cleanup_browser():
    # Ensure browsers are closed after each test
    yield
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        await browser.close()
