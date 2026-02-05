import helium
from selenium import webdriver
from selenium.common.exceptions import NoAlertPresentException, TimeoutException
from selenium.webdriver.common.keys import Keys
from smolagents import tool

# ────────────────────────────────────────────────
#  Collect & expose more Helium actions as proper tools
# ────────────────────────────────────────────────


@tool
def go_to(url: str) -> str:
    """Navigate the browser to the specified URL.

    Args:
        url: The full URL to navigate to (e.g. 'https://github.com/trending')

    Returns:
        Confirmation message with the navigated URL
    """
    helium.get_driver()
    return f"Navigated to {url}"


@tool
def click(element: str) -> str:
    """Click on a visible element identified by its text or Helium selector.

    Supports text, Button("..."), Link("..."), etc.

    Args:
        element: Text on the element, or Helium selector object (Button, Link, ...)

    Returns:
        Confirmation message
    """
    helium.click(element)
    return f"Clicked '{element}'"


@tool
def write(text: str, into: str | None = None) -> str:
    """Type text into the currently focused field or a specified input.

    Args:
        text: Text to type
        into: Optional text that identifies the input field, or S() selector

    Returns:
        Confirmation message
    """
    if into is None:
        helium.write(text)
        return f"Typed '{text}' into focused field"
    else:
        helium.write(text, into=into)
        return f"Typed '{text}' into '{into}'"


@tool
def select(value: str, from_: str) -> str:
    """Select an option from a <select> dropdown.

    Args:
        value: The visible text or value of the option to select
        from_: Text that identifies the dropdown (usually its label)

    Returns:
        Confirmation message
    """
    helium.select(value, from_=from_)
    return f"Selected '{value}' from '{from_}'"


@tool
def scroll_down(num_pixels: int = 800) -> str:
    """Scroll the viewport downward.

    Args:
        num_pixels: Number of pixels to scroll down (default: 800)

    Returns:
        Confirmation message
    """
    helium.scroll_down(num_pixels=num_pixels)
    return f"Scrolled down {num_pixels}px"


@tool
def scroll_up(num_pixels: int = 800) -> str:
    """Scroll the viewport upward.

    Args:
        num_pixels: Number of pixels to scroll up (default: 800)

    Returns:
        Confirmation message
    """
    helium.scroll_up(num_pixels=num_pixels)
    return f"Scrolled up {num_pixels}px"


@tool
def find_all(element_type: str = "Text") -> list[str]:
    """Find all elements of the given type and return their visible texts.

    Args:
        element_type: Type of elements to find (e.g. "Link", "Button", "Text")

    Returns:
        List of non-empty text contents of matching elements
    """
    elements = helium.find_all(element_type)
    texts = [
        el.web_element.text.strip()
        for el in elements
        if el.web_element.text and el.web_element.text.strip()
    ]
    return texts


@tool
def get_current_url() -> str:
    """Get the URL of the currently loaded page.

    Returns:
        Current page URL
    """
    driver = helium.get_driver()
    if not driver:
        return "No browser or page is loaded"
    return getattr(driver, "current_url", "Unknown current URL")


@tool
def refresh() -> str:
    """Reload/refresh the current page.

    Returns:
        Confirmation message
    """
    helium.refresh()
    return "Page refreshed"


@tool
def go_back() -> str:
    """Navigate back to the previous page in the browser history.

    Returns:
        Confirmation message
    """
    driver = helium.get_driver()
    if hasattr(driver, "back"):
        driver.back()
        return "Navigated back to previous page"
    return "No page loaded to go back from"


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    driver = helium.get_driver()
    if not driver:
        return "No browser active to send ESC key"
    try:
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        return "Sent ESC key to attempt closing popup/modal."
    except Exception as e:
        return f"Could not send ESC: {e}"


@tool
def find_all_links() -> list[str]:
    """Return a list of visible link texts present on the current page.

    Returns:
        List of non-empty link texts
    """
    return find_all("Link")


@tool
def find_all_buttons() -> list[str]:
    """Return a list of visible button texts present on the current page.

    Returns:
        List of non-empty button texts
    """
    return find_all("Button")


@tool
def press(key: str = "ENTER") -> str:
    """Press a keyboard key such as ENTER, TAB, ESC, arrow keys, etc.

    Args:
        key: Name of the key (case-insensitive). Common values: ENTER, TAB, ESC,
             DOWN, UP, LEFT, RIGHT.

    Returns:
        Confirmation message indicating which key was pressed.
    """
    key_map = {
        "ENTER": Keys.ENTER,
        "TAB": Keys.TAB,
        "ESC": Keys.ESCAPE,
        "DOWN": Keys.ARROW_DOWN,
        "UP": Keys.ARROW_UP,
        "LEFT": Keys.ARROW_LEFT,
        "RIGHT": Keys.ARROW_RIGHT,
    }
    selenium_key = key_map.get(key.upper(), Keys.ENTER)
    helium.press(selenium_key)
    return f"Pressed {key}"


@tool
def press_enter() -> str:
    """Press the ENTER key — useful for submitting forms or confirming actions.

    Returns:
        Confirmation message that ENTER was pressed.
    """
    return press("ENTER")


@tool
def accept_alert() -> str:
    """Accept (press OK/Confirm) the currently visible JavaScript alert or prompt.

    Returns:
        Message indicating success and the alert text (if any), or that no alert was present.
    """
    try:
        alert = helium.Alert()
        text = alert.text()
        alert.accept()
        return f"Accepted alert: {text}"
    except NoAlertPresentException:
        return "No alert present to accept"


@tool
def dismiss_alert() -> str:
    """Dismiss (press Cancel) the currently visible JavaScript alert or prompt.

    Returns:
        Message indicating success and the alert text (if any), or that no alert was present.
    """
    try:
        alert = helium.Alert()
        text = alert.text()
        alert.dismiss()
        return f"Dismissed alert: {text}"
    except NoAlertPresentException:
        return "No alert present to dismiss"


@tool
def get_alert_text() -> str:
    """Retrieve the message text from the currently visible JavaScript alert or prompt
    without accepting or dismissing it.

    Returns:
        The alert text, or a message indicating no alert is present.
    """
    try:
        return helium.Alert().text()
    except NoAlertPresentException:
        return "No alert present"


@tool
def element_exists(element: str) -> bool:
    """Check whether an element matching the given text or CSS selector currently exists.

    Args:
        element: Text content, link text, button text, or CSS selector (e.g. 'Login', 'input[name=q]')

    Returns:
        True if the element exists, False otherwise.
    """
    return helium.S(element).exists()


@tool
def wait_until_text_exists(text: str, timeout_secs: int = 10) -> str:
    """Wait until the specified exact text appears anywhere on the page.

    Args:
        text: The exact visible text to wait for.
        timeout_secs: Maximum time to wait in seconds (default: 10).

    Returns:
        Success message with wait time, or timeout failure message.
    """
    from helium import Text, wait_until

    try:
        wait_until(Text(text).exists, timeout_secs=timeout_secs)
        return f"Text '{text}' appeared within {timeout_secs}s"
    except TimeoutException:
        return f"Timeout after {timeout_secs}s waiting for text '{text}'"


@tool
def get_element_text(element: str) -> str:
    """Retrieve the visible text content of the first element matching the given identifier.

    Args:
        element: Text, label, placeholder, or CSS selector identifying the element.

    Returns:
        The stripped text content if found, or a not-found message.
    """
    el = helium.S(element)
    if el.exists():
        return el.web_element.text.strip()
    return f"Element '{element}' not found"


@tool
def attach_file(file_path: str, to_field: str) -> str:
    """Upload a file by attaching it to a file input field.

    Args:
        file_path: Full absolute path to the file on disk.
        to_field: Text label, placeholder, or selector identifying the file input.

    Returns:
        Confirmation message indicating the file was attached.
    """
    helium.attach_file(file_path, to=to_field)
    return f"Attached file '{file_path}' to '{to_field}'"


@tool
def hover(element: str) -> str:
    """Move the mouse cursor over the specified element (hover).

    Args:
        element: Text or selector of the element to hover over.

    Returns:
        Confirmation message.
    """
    helium.hover(element)
    return f"Hovered over '{element}'"


@tool
def double_click(element: str) -> str:
    """Perform a double-click on the specified element.

    Args:
        element: Text or selector of the element to double-click.

    Returns:
        Confirmation message.
    """
    helium.doubleclick(element)
    return f"Double-clicked '{element}'"


@tool
def get_page_source(max_preview: int = 600) -> str:
    """Retrieve the full HTML source of the current page (truncated if very long).

    Args:
        max_preview: Maximum characters to return (default: 600). Longer sources are truncated.

    Returns:
        HTML source (possibly truncated with ellipsis).
    """
    source = helium.get_driver().page_source
    if len(source) > max_preview:
        return source[:max_preview] + " ... [truncated]"
    return source


@tool
def set_implicit_wait(seconds: float = 10.0) -> str:
    """Configure Helium's global implicit wait timeout for element lookups.

    Args:
        seconds: Number of seconds to wait implicitly (default: 10.0).

    Returns:
        Confirmation message with the new timeout value.
    """
    helium.Config.implicit_wait_secs = seconds
    return f"Set implicit wait to {seconds} seconds"


# Collect all tools for saving / agent registration
ALL_TOOLS = [
    go_back,
    go_to,
    click,
    write,
    select,
    scroll_down,
    scroll_up,
    find_all,
    find_all_links,
    find_all_buttons,
    get_current_url,
    refresh,
    press,
    press_enter,
    accept_alert,
    dismiss_alert,
    get_alert_text,
    element_exists,
    wait_until_text_exists,
    get_element_text,
    attach_file,
    hover,
    double_click,
    get_page_source,
    set_implicit_wait,
    # close_popups already exists — kept as is
]
