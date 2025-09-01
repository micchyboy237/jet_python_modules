import pytest
import logging
from jet.libs.autogen.playwright_controller import PlaywrightController
from playwright.async_api import async_playwright
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

TEST_LINKS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Links Page</title>
</head>
<body>
    <a href="https://example.com">Example Site</a>
    <a href="/about">About Page</a>
    <a href="/contact">Contact Page</a>
    <a href="javascript:void(0)">Invalid Link</a>
    <a href="https://hidden.com" style="display: none;">Hidden Link</a>
</body>
</html>
"""

TEST_FORM_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Test Form Page</title>
</head>
<body>
    <form id="form1">
        <label for="username">Username:</label>
        <input type="text" id="username" aria-label="Enter username">
        <textarea id="bio"></textarea>
        <button id="submit-btn">Submit Form</button>
    </form>
    <form id="form2">
        <label for="color">Choose color:</label>
        <select id="color" aria-label="Select favorite color">
            <option value="red">Red</option>
            <option value="blue">Blue</option>
        </select>
    </form>
    <input type="text" id="hidden-input" style="display: none;">
</body>
</html>
"""

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
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        start_time = await page.evaluate("() => performance.now()")
        await controller.sleep(page, 0.5)
        end_time = await page.evaluate("() => performance.now()")
        elapsed_time = end_time - start_time
        expected_min_time = 500
        assert elapsed_time >= expected_min_time, f"Expected sleep time >= {expected_min_time}ms, but got {elapsed_time}ms"


@pytest.mark.asyncio
async def test_playwright_controller_get_visual_viewport() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)
        result = await controller.get_visual_viewport(page)
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
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        rects = await controller.get_interactive_rects(page)
        input_box_id = next(
            (rect for rect in rects if rects[rect]["tag_name"] == "input, type=text"), None)
        await page.locator(f"[__elementId='{input_box_id}']").focus()
        result = await controller.get_focused_rect_id(page)
        expected_id = input_box_id
        assert result == expected_id, f"Expected focused rect ID {expected_id}, got {result}"


@pytest.mark.asyncio
async def test_playwright_controller_get_page_metadata() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        result = await controller.get_page_metadata(page)
        logger.debug(f"Page metadata result: {result}")
        expected_metadata = {
            "meta_tags": {
                "viewport": "width=device-width, initial-scale=1.0"
            }
        }
        assert isinstance(result, dict), "Metadata is not a dictionary"
        assert result.get(
            "meta_tags") == expected_metadata["meta_tags"], f"Expected meta_tags {expected_metadata['meta_tags']}, got {result.get('meta_tags')}"


@pytest.mark.asyncio
async def test_playwright_controller_back() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto("data:text/html,<h1>First Page</h1>")
        await page.wait_for_load_state()
        await page.goto("data:text/html," + FAKE_HTML)
        await page.wait_for_load_state()
        controller = PlaywrightController()
        await controller.back(page)
        expected_title = "First Page"
        result_title = await page.evaluate("document.querySelector('h1').innerText")
        assert result_title == expected_title, f"Expected title {expected_title}, got {result_title}"


@pytest.mark.asyncio
async def test_playwright_controller_page_down() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)
        initial_scroll = await page.evaluate("window.scrollY")
        await controller.page_down(page)
        result_scroll = await page.evaluate("window.scrollY")
        expected_scroll = initial_scroll + (900 - 50)
        assert result_scroll == expected_scroll, f"Expected scroll position {expected_scroll}, got {result_scroll}"


@pytest.mark.asyncio
async def test_playwright_controller_page_up() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(
            viewport_width=1440, viewport_height=900)
        await page.evaluate("window.scrollTo(0, 1000);")
        initial_scroll = await page.evaluate("window.scrollY")
        await controller.page_up(page)
        result_scroll = await page.evaluate("window.scrollY")
        expected_scroll = initial_scroll - (900 - 50)
        assert result_scroll == expected_scroll, f"Expected scroll position {expected_scroll}, got {result_scroll}"


@pytest.mark.asyncio
async def test_playwright_controller_hover_id() -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController(animate_actions=False)
        rects = await controller.get_interactive_rects(page)
        expected_button_id = next(
            (rect for rect in rects if rects[rect]["aria_name"] == "Click Me"), None)
        expected_is_visible = True
        await controller.hover_id(page, expected_button_id)
        result_is_visible = await page.locator(f"[__elementId='{expected_button_id}']").is_visible()
        logger.debug(
            f"Hovered element visibility: {result_is_visible}, expected: {expected_is_visible}")
        assert result_is_visible == expected_is_visible, f"Expected button visibility {expected_is_visible}, got {result_is_visible}"


@pytest.mark.asyncio
async def test_playwright_controller_search_bar_submit() -> None:
    """
    Test that fill_id can find a search bar by ID, input text, and submit with Enter.
    Given: A page with a text input acting as a search bar
    When: The fill_id method is called with the search bar's ID and a query
    Then: The input contains the query and the element remains focused after Enter
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(FAKE_HTML)
        controller = PlaywrightController()
        rects = await controller.get_interactive_rects(page)
        search_bar_id = next(
            (rect for rect in rects if rects[rect]
             ["tag_name"] == "input, type=text"),
            None
        )
        assert search_bar_id is not None, "Search bar input not found in interactive rects"
        search_query = "example search"
        await controller.fill_id(page, search_bar_id, search_query, press_enter=True)
        result_value = await page.evaluate("document.getElementById('input-box').value")
        expected_value = search_query
        assert result_value == expected_value, f"Expected input value '{expected_value}', got '{result_value}'"
        result_focused_id = await controller.get_focused_rect_id(page)
        expected_focused_id = search_bar_id
        assert result_focused_id == expected_focused_id, f"Expected focused ID '{expected_focused_id}', got '{result_focused_id}'"


@pytest.mark.asyncio
async def test_playwright_controller_get_form_targets() -> None:
    """
    Test that get_interactive_rects returns a list of form target IDs and texts.
    Given: A page with diverse form elements including inputs, textarea, select, and button
    When: The get_interactive_rects method is called
    Then: The result includes the IDs and texts of visible form-related elements
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(TEST_FORM_HTML)
        controller = PlaywrightController()

        # When: Retrieve interactive rects
        rects = await controller.get_interactive_rects(page)

        # Then: Process rects to get form targets (inputs, buttons, textareas, selects)
        result = [
            {"id": rect_id, "text": rects[rect_id]["aria_name"]}
            for rect_id in rects
            if rects[rect_id]["tag_name"].startswith(("input", "button", "textarea", "select"))
        ]

        # Expected: List of form targets with their IDs and texts
        expected = [
            {"id": any, "text": "Enter username"},
            {"id": any, "text": ""},
            {"id": any, "text": "Submit Form"},
            {"id": any, "text": "Select favorite color"}
        ]

        # Assert each expected item exists in the result (ignoring specific IDs)
        for expected_item in expected:
            assert any(
                r["text"] == expected_item["text"] for r in result
            ), f"Expected form target with text '{expected_item['text']}' not found in {result}"

        # Verify the total number of form targets
        assert len(
            result) == 4, f"Expected 4 form targets, got {len(result)}: {result}"


@pytest.mark.asyncio
async def test_playwright_controller_get_links() -> None:
    """
    Test that get_links returns a list of hyperlinks with their text and href, with optional visibility filtering.
    Given: A page with visible, hidden, and invalid links
    When: The get_links method is called with visible_only=True and visible_only=False
    Then: The result includes the expected links based on visibility
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.set_content(TEST_LINKS_HTML)
        controller = PlaywrightController()

        # When: Retrieve visible links only
        base_url = "http://test.com"
        result_visible = await controller.get_links(page, base_url, visible_only=True)

        # Then: Verify only visible, valid links are returned
        expected_visible = [
            {"text": "Example Site", "href": "https://example.com/"},
            {"text": "About Page", "href": "http://test.com/about"},
            {"text": "Contact Page", "href": "http://test.com/contact"}
        ]
        result_visible_sorted = sorted(result_visible, key=lambda x: x["text"])
        expected_visible_sorted = sorted(
            expected_visible, key=lambda x: x["text"])
        assert result_visible_sorted == expected_visible_sorted, f"Expected visible links {expected_visible_sorted}, got {result_visible_sorted}"
        assert len(
            result_visible) == 3, f"Expected 3 visible links, got {len(result_visible)}: {result_visible}"

        # When: Retrieve all links (visible and hidden)
        result_all = await controller.get_links(page, base_url, visible_only=False)

        # Then: Verify all valid links are returned, including hidden
        expected_all = [
            {"text": "Example Site", "href": "https://example.com/"},
            {"text": "About Page", "href": "http://test.com/about"},
            {"text": "Contact Page", "href": "http://test.com/contact"},
            {"text": "Hidden Link", "href": "https://hidden.com/"}
        ]
        result_all_sorted = sorted(result_all, key=lambda x: x["text"])
        expected_all_sorted = sorted(expected_all, key=lambda x: x["text"])
        assert result_all_sorted == expected_all_sorted, f"Expected all links {expected_all_sorted}, got {result_all_sorted}"
        assert len(
            result_all) == 4, f"Expected 4 links (including hidden), got {len(result_all)}: {result_all}"


@pytest.fixture(autouse=True)
async def cleanup_browser():
    yield
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        await browser.close()
