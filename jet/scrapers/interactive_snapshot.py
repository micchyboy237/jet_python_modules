import base64
import json
import os
import time
from pathlib import Path
from typing import List, Optional, TypedDict
from urllib.parse import urlparse

from fake_useragent import UserAgent
from jet.logger import logger
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


class ElementCoordinates(TypedDict):
    """Coordinates and dimensions of an element."""

    selector: str
    x: float
    y: float
    width: float
    height: float
    tag: str
    text: str


class InteractiveScreenshotResult(TypedDict):
    """Result of taking an interactive element screenshot."""

    url: str
    screenshot_path: str
    screenshot_base64: str
    aria_snapshot_path: str
    aria_snapshot_content: str
    coordinates_path: str
    coordinates: List[ElementCoordinates]
    highlighted_elements_count: int
    element_selectors: List[str]


def highlight_interactive_elements_and_screenshot(
    url: str,
    output_dir: str = "generated",
    timeout_ms: int = 10000,
    headless: bool = True,
    highlight_color: str = "#FF0000",
    highlight_border_width: int = 3,
    include_buttons: bool = True,
    include_links: bool = True,
    include_inputs: bool = True,
    custom_selectors: Optional[List[str]] = None,
    aria_snapshot_mode: str = "ai",
    include_aria_boxes: bool = True,
) -> InteractiveScreenshotResult:
    """
    Opens a URL in a browser, highlights all interactive elements, takes a screenshot,
    saves an ARIA snapshot, and extracts precise coordinates for each interactive element.

    Args:
        url: The URL to visit.
        output_dir: Directory to save artifacts (defaults to "generated").
        timeout_ms: Timeout for page loading in milliseconds.
        headless: Whether to run browser in headless mode.
        highlight_color: CSS color for highlighting borders.
        highlight_border_width: Width of highlight border in pixels.
        include_buttons: Whether to highlight button elements.
        include_links: Whether to highlight link elements.
        include_inputs: Whether to highlight input/textarea/select elements.
        custom_selectors: Additional CSS selectors to highlight.
        aria_snapshot_mode: Mode for ARIA snapshot ("ai" or "default").
        include_aria_boxes: Whether to include bounding boxes in the ARIA snapshot YAML.

    Returns:
        InteractiveScreenshotResult with paths and metadata.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build CSS selector for interactive elements
    selectors = []
    if include_buttons:
        selectors.append("button")
        selectors.append("[role='button']")
        selectors.append("[role='link']")
        selectors.append("[role='tab']")
        selectors.append("[role='menuitem']")
    if include_links:
        selectors.append("a[href]")
    if include_inputs:
        selectors.append("input")
        selectors.append("textarea")
        selectors.append("select")
    if custom_selectors:
        selectors.extend(custom_selectors)

    combined_selector = ", ".join(selectors) if selectors else "*"

    ua = UserAgent()
    screenshot_path = ""
    screenshot_base64 = ""
    aria_snapshot_path = ""
    aria_snapshot_content = ""
    coordinates_path = ""
    coordinates_list: List[ElementCoordinates] = []
    highlighted_count = 0
    element_selectors_list: List[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        )
        page = browser.new_page(user_agent=ua.random)

        try:
            page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout loading {url}, proceeding with available content")

        # Wait for page to stabilize
        page.wait_for_timeout(1000)

        # JavaScript to highlight elements and extract their coordinates
        js_code = """
        (config) => {
            const elements = document.querySelectorAll(config.selector);
            const results = [];
            
            elements.forEach((el) => {
                // Skip hidden or disabled elements
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden') {
                    return;
                }
                
                // Get bounding box
                const rect = el.getBoundingClientRect();
                
                // Generate a unique selector
                let selector = '';
                if (el.id) {
                    selector = '#' + el.id;
                } else if (el.className && typeof el.className === 'string') {
                    const classes = el.className.split(' ').filter(c => c && !c.startsWith('css-')).slice(0, 2);
                    if (classes.length > 0) {
                        selector = el.tagName.toLowerCase() + '.' + classes.join('.');
                    } else {
                        selector = el.tagName.toLowerCase();
                    }
                } else {
                    selector = el.tagName.toLowerCase();
                }
                
                // Add nth-of-type if needed
                const parent = el.parentElement;
                if (parent) {
                    const siblings = Array.from(parent.children).filter(sib => sib.tagName === el.tagName);
                    if (siblings.length > 1) {
                        const idx = siblings.indexOf(el) + 1;
                        selector += ':nth-of-type(' + idx + ')';
                    }
                }
                
                // Apply highlight
                const originalOutline = el.style.outline;
                el.style.outline = config.width + 'px solid ' + config.color;
                el.style.outlineOffset = '2px';
                el.dataset.originalOutline = originalOutline;
                el.dataset.highlighted = 'true';
                
                results.push({
                    selector: selector,
                    x: rect.x,
                    y: rect.y,
                    width: rect.width,
                    height: rect.height,
                    tag: el.tagName.toLowerCase(),
                    text: (el.innerText || el.value || '').trim().substring(0, 100)
                });
            });
            
            window._highlightedSelectors = results.map(r => r.selector);
            return results;
        }
        """

        # Execute JS and get coordinates
        element_data = page.evaluate(
            js_code,
            {
                "selector": combined_selector,
                "color": highlight_color,
                "width": highlight_border_width,
            },
        )

        highlighted_count = len(element_data)
        element_selectors_list = [item["selector"] for item in element_data]
        coordinates_list = element_data

        # Generate filenames based on URL and timestamp
        parsed_url = urlparse(url)
        timestamp = int(time.time())
        safe_host = parsed_url.netloc.replace(".", "_")

        # Take screenshot
        screenshot_filename = f"{safe_host}_{timestamp}_interactive.png"
        screenshot_path = os.path.join(output_dir, screenshot_filename)

        try:
            screenshot_bytes = page.screenshot(full_page=True, path=screenshot_path)
            if screenshot_bytes:
                screenshot_base64 = base64.b64encode(screenshot_bytes).decode("utf-8")
                logger.success(f"Screenshot saved: {screenshot_path}")
        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            raise

        # Capture ARIA snapshot
        try:
            aria_snapshot_content = page.aria_snapshot(
                mode=aria_snapshot_mode, boxes=include_aria_boxes
            )

            aria_snapshot_filename = f"{safe_host}_{timestamp}_snapshot.yml"
            aria_snapshot_path = os.path.join(output_dir, aria_snapshot_filename)

            with open(aria_snapshot_path, "w", encoding="utf-8") as f:
                f.write(aria_snapshot_content)

            logger.success(f"ARIA snapshot saved: {aria_snapshot_path}")
        except Exception as e:
            logger.warning(f"Failed to capture ARIA snapshot: {e}")

        # Save coordinates as JSON
        if coordinates_list:
            coordinates_filename = f"{safe_host}_{timestamp}_coords.json"
            coordinates_path = os.path.join(output_dir, coordinates_filename)

            with open(coordinates_path, "w", encoding="utf-8") as f:
                json.dump(coordinates_list, f, indent=2)

            logger.success(f"Element coordinates saved: {coordinates_path}")

        browser.close()

    return InteractiveScreenshotResult(
        url=url,
        screenshot_path=screenshot_path,
        screenshot_base64=screenshot_base64,
        aria_snapshot_path=aria_snapshot_path,
        aria_snapshot_content=aria_snapshot_content,
        coordinates_path=coordinates_path,
        coordinates=coordinates_list,
        highlighted_elements_count=highlighted_count,
        element_selectors=element_selectors_list,
    )


if __name__ == "__main__":
    result = highlight_interactive_elements_and_screenshot(
        url="https://ph.indeed.com/jobs?q=Machine+Learning+Engineer&l=Remote&from=searchOnDesktopSerp&vjk=b8cc43c29a4ae42f",
        headless=False,
        output_dir="generated",
        highlight_color="#FF0000",
    )
    print(f"Highlighted {result['highlighted_elements_count']} elements")
    print(f"Screenshot: {result['screenshot_path']}")
    print(f"ARIA Snapshot: {result['aria_snapshot_path']}")
    print(f"Coordinates: {result['coordinates_path']}")
