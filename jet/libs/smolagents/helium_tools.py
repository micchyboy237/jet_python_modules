from typing import List, Optional
import helium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from smolagents import CodeAgent, tool, InferenceClientModel

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
    helium.go_to(url)
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
def write(text: str, into: Optional[str] = None) -> str:
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
def find_all(element_type: str = "Text") -> List[str]:
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
    return helium.get_driver().current_url


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
    helium.get_driver().back()
    return "Navigated back to previous page"


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    webdriver.ActionChains(helium.get_driver()).send_keys(Keys.ESCAPE).perform()
    return "Sent ESC key to attempt closing popup/modal."


@tool
def find_all_links() -> List[str]:
    """Return a list of visible link texts present on the current page.

    Returns:
        List of non-empty link texts
    """
    return find_all("Link")


@tool
def find_all_buttons() -> List[str]:
    """Return a list of visible button texts present on the current page.

    Returns:
        List of non-empty button texts
    """
    return find_all("Button")


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
    # ← add more wrappers here in the future
]
