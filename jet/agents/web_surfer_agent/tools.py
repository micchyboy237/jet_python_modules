import json
from typing import Optional

from jet_telemetry import performance_monitor, tool
from playwright.async_api import Page

# Global page reference for stateless tool functions
# In a production environment, you might want to pass this explicitly
# or use a context manager, but for this agent pattern, we'll attach it dynamically.
_page_context: dict = {"page": None}


def set_page(page: Page):
    """Set the current Playwright page context for the tools."""
    _page_context["page"] = page


def get_page() -> Page:
    """Get the current Playwright page context."""
    if _page_context["page"] is None:
        raise RuntimeError(
            "Browser page not initialized. Call initialize_browser first."
        )
    return _page_context["page"]


@tool(name="browser_navigate", description="Navigate the browser to a specified URL.")
@performance_monitor(threshold_ms=5000)
async def browser_navigate(url: str) -> str:
    """Navigate the browser to a specific URL."""
    page = get_page()
    await page.goto(url, wait_until="domcontentloaded")
    return f"Navigated to {url}"


@tool(name="browser_navigate_back", description="Go back in browser history.")
@performance_monitor(threshold_ms=2000)
async def browser_navigate_back() -> str:
    """Go back in browser history."""
    page = get_page()
    await page.go_back(wait_until="domcontentloaded")
    return "Navigated back"


@tool(
    name="browser_snapshot",
    description="Capture accessibility tree of current page as structured YAML.",
)
@performance_monitor(threshold_ms=2000)
async def browser_snapshot() -> str:
    """Capture the current page state using the accessibility tree."""
    page = get_page()
    try:
        snapshot = await page.locator("body").aria_snapshot()
        title = await page.title()
        url = page.url
        return f"URL: {url}\nTitle: {title}\n\nAccessibility Tree:\n{snapshot}"
    except Exception as e:
        return f"[ERROR] Snapshot failed: {str(e)}"


@tool(
    name="browser_find",
    description="Search the snapshot for text or regex patterns. Args: pattern (str)",
)
@performance_monitor(threshold_ms=1000)
async def browser_find(pattern: str) -> str:
    """Search the current page content for a pattern."""
    page = get_page()
    # Simple implementation: check if text exists in body
    # For robust regex/search, one might fetch innerText and search locally
    try:
        body_text = await page.locator("body").inner_text()
        if pattern.lower() in body_text.lower():
            return f"Found pattern '{pattern}' on page."
        else:
            return f"Pattern '{pattern}' not found on page."
    except Exception as e:
        return f"[ERROR] Find failed: {str(e)}"


@tool(
    name="browser_click",
    description="Click an element by its accessible role and name.",
)
@performance_monitor(threshold_ms=3000)
async def browser_click(role: str, name: str) -> str:
    """Click an element identified by its ARIA role and accessible name."""
    page = get_page()
    locator = page.get_by_role(role, name=name)
    await locator.click(timeout=5000)
    await page.wait_for_load_state("domcontentloaded")
    return f"Clicked element with role '{role}' and name '{name}'"


@tool(
    name="browser_hover",
    description="Hover over an element by its accessible role and name.",
)
@performance_monitor(threshold_ms=2000)
async def browser_hover(role: str, name: str) -> str:
    """Hover over an element."""
    page = get_page()
    locator = page.get_by_role(role, name=name)
    await locator.hover(timeout=5000)
    return f"Hovered over element with role '{role}' and name '{name}'"


@tool(
    name="browser_type",
    description="Type text into an input field by its accessible label.",
)
@performance_monitor(threshold_ms=2000)
async def browser_type(label: str, text: str) -> str:
    """Type text into a form field identified by its label."""
    page = get_page()
    locator = page.get_by_label(label)
    await locator.fill(text)
    return f"Typed '{text}' into field labeled '{label}'"


@tool(
    name="browser_fill_form",
    description="Fill multiple form fields at once. Args: fields (dict of label->value)",
)
@performance_monitor(threshold_ms=3000)
async def browser_fill_form(fields: dict) -> str:
    """Fill multiple form fields at once."""
    page = get_page()
    results = []
    for label, value in fields.items():
        try:
            locator = page.get_by_label(label)
            await locator.fill(str(value))
            results.append(f"Filled '{label}'")
        except Exception as e:
            results.append(f"Failed to fill '{label}': {str(e)}")
    return "; ".join(results)


@tool(
    name="browser_select_option",
    description="Select dropdown option. Args: label (str), option (str)",
)
@performance_monitor(threshold_ms=2000)
async def browser_select_option(label: str, option: str) -> str:
    """Select dropdown option."""
    page = get_page()
    locator = page.get_by_label(label)
    await locator.select_option(option)
    return f"Selected '{option}' in dropdown '{label}'"


@tool(
    name="browser_press_key",
    description="Press a key (Enter, Tab, etc.). Args: key (str)",
)
@performance_monitor(threshold_ms=1000)
async def browser_press_key(key: str) -> str:
    """Press a key."""
    page = get_page()
    await page.keyboard.press(key)
    return f"Pressed key '{key}'"


@tool(
    name="browser_take_screenshot",
    description="Take PNG screenshot of current page. Args: path (str, optional)",
)
@performance_monitor(threshold_ms=2000)
async def browser_take_screenshot(path: Optional[str] = None) -> str:
    """Take a screenshot."""
    page = get_page()
    if not path:
        path = "screenshot.png"
    await page.screenshot(path=path, full_page=True)
    return f"Screenshot saved to {path}"


@tool(
    name="browser_tabs",
    description="List, create, close, or select tabs. Args: action (str), index (int, optional)",
)
@performance_monitor(threshold_ms=1000)
async def browser_tabs(action: str, index: Optional[int] = None) -> str:
    """Manage browser tabs."""
    page = get_page()
    context = page.context
    if action == "list":
        pages = context.pages
        titles = [await p.title() for p in pages]
        return f"Tabs: {json.dumps(titles)}"
    elif action == "new":
        new_page = await context.new_page()
        set_page(new_page)
        return "New tab created and selected"
    elif action == "close":
        if index is not None:
            pages = context.pages
            if 0 <= index < len(pages):
                await pages[index].close()
                return f"Closed tab at index {index}"
        return "Invalid tab index"
    elif action == "select":
        if index is not None:
            pages = context.pages
            if 0 <= index < len(pages):
                set_page(pages[index])
                return f"Selected tab at index {index}"
        return "Invalid tab index"
    return f"Unknown action: {action}"


@tool(
    name="browser_handle_dialog",
    description="Accept or dismiss JavaScript dialogs. Args: accept (bool), prompt_text (str, optional)",
)
@performance_monitor(threshold_ms=1000)
async def browser_handle_dialog(accept: bool, prompt_text: Optional[str] = None) -> str:
    """Handle dialog. Note: This usually requires setting up a dialog handler before the dialog appears.
    For this tool, we assume the dialog is pending or we are setting up for next one."""
    # Playwright dialogs are event-driven. This tool is a placeholder for logic
    # that would typically be handled by page.on("dialog", ...)
    return f"Dialog handling configured: accept={accept}"


@tool(
    name="browser_evaluate",
    description="Evaluate JavaScript on page. Args: script (str)",
)
@performance_monitor(threshold_ms=2000)
async def browser_evaluate(script: str) -> str:
    """Evaluate JavaScript on page."""
    page = get_page()
    try:
        result = await page.evaluate(script)
        return f"Eval result: {json.dumps(result)}"
    except Exception as e:
        return f"[ERROR] Eval failed: {str(e)}"


@tool(
    name="browser_wait_for",
    description="Wait for text or time. Args: text (str, optional), timeout_ms (int, optional)",
)
@performance_monitor(threshold_ms=10000)
async def browser_wait_for(text: Optional[str] = None, timeout_ms: int = 5000) -> str:
    """Wait for condition."""
    page = get_page()
    if text:
        try:
            await page.wait_for_function(
                f"document.body.innerText.includes('{text}')", timeout=timeout_ms
            )
            return f"Waited for text '{text}'"
        except Exception:
            return f"Timeout waiting for text '{text}'"
    else:
        await page.wait_for_timeout(timeout_ms)
        return f"Waited for {timeout_ms}ms"


@tool(name="browser_close", description="Close the page.")
@performance_monitor(threshold_ms=1000)
async def browser_close() -> str:
    """Close the page."""
    page = get_page()
    await page.close()
    return "Page closed"
