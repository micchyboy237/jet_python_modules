import pytest
import asyncio
from playwright.async_api import async_playwright
from jet.libs.autogen.playwright_controller import PlaywrightController


@pytest.fixture
async def page():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        yield page
        await context.close()
        await browser.close()


@pytest.fixture
def controller():
    return PlaywrightController(downloads_folder=None, animate_actions=False)


class TestPlaywrightController:
    @pytest.mark.asyncio
    async def test_find_form_elements_by_type(self, page, controller):
        # Given: A page with a text input
        await page.set_content('<input type="text" __elementId="1" name="username">')

        # When: Finding elements of type 'textbox'
        result = await controller.find_form_elements(page, element_type="textbox")

        # Then: The correct element is found
        expected = [{"id": "1", "type": "text",
                     "name": "username", "role": ""}]
        assert result == expected, f"Expected {expected}, but got {result}"

    @pytest.mark.asyncio
    async def test_find_form_elements_by_name(self, page, controller):
        # Given: A page with a button with a specific name
        await page.set_content('<button __elementId="2" name="submit">Click me</button>')

        # When: Finding elements by name 'submit'
        result = await controller.find_form_elements(page, name="submit")

        # Then: The correct element is found
        expected = [{"id": "2", "type": "button",
                     "name": "submit", "role": ""}]
        assert result == expected, f"Expected {expected}, but got {result}"

    @pytest.mark.asyncio
    async def test_select_option(self, page, controller):
        # Given: A page with a dropdown
        await page.set_content('''
            <select __elementId="3">
                <option value="opt1">Option 1</option>
                <option value="opt2">Option 2</option>
            </select>
        ''')

        # When: Selecting an option by value
        await controller.select_option(page, "3", "opt2")

        # Then: The correct option is selected
        selected = await page.evaluate('() => document.querySelector("select").value')
        expected = "opt2"
        assert selected == expected, f"Expected selected value {expected}, but got {selected}"

    @pytest.mark.asyncio
    async def test_submit_form(self, page, controller):
        # Given: A page with a form and a text input
        await page.set_content('''
            <form action="javascript:alert('submitted')">
                <input type="text" __elementId="4" name="input">
            </form>
        ''')

        # When: Submitting the form via an element
        async with page.expect_popup() as popup_info:
            await controller.submit_form(page, "4")

        # Then: The form is submitted (popup triggered)
        popup = await popup_info.value
        assert popup is not None, "Expected form submission to trigger a popup"
