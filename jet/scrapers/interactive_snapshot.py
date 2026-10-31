import base64
import json
import os
import time
from pathlib import Path
from typing import List, TypedDict
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from jet.logger import logger
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


class ElementCoordinates(TypedDict):
    """Coordinates and dimensions of an element."""

    selector: str
    xpath: str  # Full path starting from root using only tags
    x: float
    y: float
    width: float
    height: float
    tag: str
    text: str
    events: List[str]


class InteractiveScreenshotResult(TypedDict):
    """Result of taking an interactive element screenshot."""

    url: str
    screenshot_path: str
    screenshot_base64: str
    aria_snapshot_path: str
    aria_snapshot_content: str
    coordinates_path: str
    coordinates: List[ElementCoordinates]
    interactive_elements_path: str
    highlighted_elements_count: int
    element_selectors: List[str]


def highlight_interactive_elements_and_screenshot(
    url: str,
    output_dir: str = "generated",
    timeout_ms: int = 15000,
    headless: bool = True,
    highlight_color: str = "#FF0000",
    highlight_border_width: int = 3,
    aria_snapshot_mode: str = "ai",
    include_aria_boxes: bool = True,
    root_selector: str = "html",  # CSS selector for root element
) -> InteractiveScreenshotResult:
    """
    Opens a URL, highlights all interactive elements (including divs with JS listeners),
    takes a screenshot, saves an ARIA snapshot, and extracts precise coordinates.

    Args:
        url: The URL to scrape
        output_dir: Directory to save output files
        timeout_ms: Page load timeout in milliseconds
        headless: Whether to run browser in headless mode
        highlight_color: Color for highlighting interactive elements
        highlight_border_width: Width of highlight border in pixels
        aria_snapshot_mode: Mode for ARIA snapshot generation
        include_aria_boxes: Whether to include bounding boxes in ARIA snapshot
        root_selector: CSS selector for root element. All interactive elements
                      will be searched within this root. Default is "html".
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    ua = UserAgent()
    screenshot_path = ""
    screenshot_base64 = ""
    aria_snapshot_path = ""
    aria_snapshot_content = ""
    coordinates_path = ""
    interactive_elements_path = ""
    coordinates_list: List[ElementCoordinates] = []
    highlighted_count = 0
    element_selectors_list: List[str] = []

    # Advanced JS detection using Event Listener Monkey-patching
    detection_script = """
    (rootSelector) => {
        const config = {
            color: "%s",
            width: %d
        };

        // Function to generate XPath for an element relative to root
        // Only uses tag names with indices, no IDs or classes
        function getXPath(element, rootElement) {
            if (element === rootElement) {
                return element.tagName.toLowerCase();
            }
            
            if (!element || !element.parentNode) {
                return '';
            }
            
            // Check if we've reached above the root
            if (!rootElement.contains(element)) {
                return '';
            }
            
            let ix = 0;
            const siblings = element.parentNode.childNodes;
            
            for (let i = 0; i < siblings.length; i++) {
                const sibling = siblings[i];
                
                if (sibling === element) {
                    const parentXPath = getXPath(element.parentNode, rootElement);
                    if (parentXPath === '') return '';
                    return parentXPath + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
                }
                
                if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                    ix++;
                }
            }
            
            return '';
        }

        function getTrackedEvents(el) {
            const events = [];
            if (el.hasAttribute('onclick')) events.push('onclick (inline)');
            
            if (window._trackedEvents && window._trackedEvents.has(el)) {
                const tracked = window._trackedEvents.get(el);
                ['click', 'mousedown', 'mouseup', 'touchstart', 'keydown', 'keyup'].forEach(evt => {
                    if (tracked.has(evt)) events.push(evt);
                });
            }
            return events;
        }

        function isInteractive(el) {
            const interactiveTags = ['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'SUMMARY'];
            if (interactiveTags.includes(el.tagName)) return true;
            
            const role = el.getAttribute('role');
            if (role && ['button', 'link', 'tab', 'menuitem', 'option', 'checkbox'].includes(role)) return true;
            
            if (el.hasAttribute('tabindex') && el.getAttribute('tabindex') !== '-1') return true;
            if (el.hasAttribute('onclick')) return true;
            
            // Check our tracked events
            if (window._trackedEvents && window._trackedEvents.has(el)) {
                const tracked = window._trackedEvents.get(el);
                if (tracked.has('click') || tracked.has('mousedown') || tracked.has('touchstart')) return true;
            }
            
            return false;
        }

        function processElement(el, rootElement) {
            if (el.dataset.highlighted === 'true') return null;
            if (!isInteractive(el)) return null;
            
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return null;

            // Highlight
            el.style.outline = `${config.width}px solid ${config.color}`;
            el.style.outlineOffset = '2px';
            el.dataset.highlighted = 'true';

            // Generate Selector (relative to document, can use id/class for readability)
            let selector = el.tagName.toLowerCase();
            if (el.id) selector += '#' + el.id;
            else if (el.className) {
                const firstClass = el.className.split(' ')[0];
                if (firstClass) selector += '.' + firstClass;
            }

            // Generate XPath relative to root - ONLY uses tags with indices
            const xpath = getXPath(el, rootElement);

            const rect = el.getBoundingClientRect();
            return {
                selector: selector,
                xpath: xpath,
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height),
                tag: el.tagName.toLowerCase(),
                text: (el.innerText || el.value || '').trim().substring(0, 50),
                events: getTrackedEvents(el)
            };
        }

        // Find root element
        let rootElement;
        if (rootSelector === 'html' || rootSelector === 'body') {
            rootElement = document.querySelector(rootSelector);
        } else {
            rootElement = document.querySelector(rootSelector);
        }
        
        if (!rootElement) {
            console.warn(`Root element not found for selector: ${rootSelector}`);
            rootElement = document.documentElement;
        }

        // Process existing elements within root
        const results = [];
        rootElement.querySelectorAll('*').forEach(el => {
            const data = processElement(el, rootElement);
            if (data) results.push(data);
        });

        // Observe future elements within root
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1 && rootElement.contains(node)) {
                        processElement(node, rootElement);
                        node.querySelectorAll('*').forEach(child => {
                            if (rootElement.contains(child)) {
                                processElement(child, rootElement);
                            }
                        });
                    }
                });
            });
        });
        observer.observe(rootElement, { childList: true, subtree: true });

        return results;
    }
    """ % (highlight_color, highlight_border_width)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
        )
        page = browser.new_page(user_agent=ua.random)

        # Inject the monkey-patch BEFORE navigation so we catch early events
        page.add_init_script("""
            window._trackedEvents = new WeakMap();
            const originalAddEventListener = EventTarget.prototype.addEventListener;
            EventTarget.prototype.addEventListener = function(type, listener, options) {
                if (!window._trackedEvents.has(this)) {
                    window._trackedEvents.set(this, new Set());
                }
                window._trackedEvents.get(this).add(type);
                return originalAddEventListener.call(this, type, listener, options);
            };
        """)

        try:
            page.goto(url, timeout=timeout_ms, wait_until="networkidle")
        except PlaywrightTimeoutError:
            logger.warning(f"Timeout loading {url}, proceeding with available content")

        # Reduced wait for final JS rendering
        page.wait_for_timeout(500)

        # Execute the detection script with root selector
        element_data = page.evaluate(detection_script, root_selector)

        highlighted_count = len(element_data)
        element_selectors_list = [item["selector"] for item in element_data]
        coordinates_list = element_data

        # Get the HTML content and process it with BeautifulSoup based on root
        try:
            html_content = page.content()
            soup = BeautifulSoup(html_content, "html.parser")

            # Find root element in parsed HTML
            if root_selector == "html":
                root_element = soup.html
            elif root_selector == "body":
                root_element = soup.body
            else:
                root_element = soup.select_one(root_selector)

            if root_element:
                # Extract only the root element's HTML
                filtered_html = str(root_element)
            else:
                logger.warning(
                    f"Root element '{root_selector}' not found in HTML, using full HTML"
                )
                filtered_html = html_content
        except Exception as e:
            logger.warning(f"Failed to process HTML with BeautifulSoup: {e}")
            filtered_html = html_content

        # Generate filenames
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

        # Save interactive elements summary
        if element_data:
            interactive_filename = f"{safe_host}_{timestamp}_elements.json"
            interactive_elements_path = os.path.join(output_dir, interactive_filename)

            with open(interactive_elements_path, "w", encoding="utf-8") as f:
                json.dump(element_data, f, indent=2)

            logger.success(
                f"Interactive elements summary saved: {interactive_elements_path}"
            )

        # Save filtered HTML based on root selector
        html_filename = f"{safe_host}_{timestamp}_filtered.html"
        html_path = os.path.join(output_dir, html_filename)

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(filtered_html)

        logger.success(f"Filtered HTML saved: {html_path}")

        browser.close()

    return InteractiveScreenshotResult(
        url=url,
        screenshot_path=screenshot_path,
        screenshot_base64=screenshot_base64,
        aria_snapshot_path=aria_snapshot_path,
        aria_snapshot_content=aria_snapshot_content,
        coordinates_path=coordinates_path,
        coordinates=coordinates_list,
        interactive_elements_path=interactive_elements_path,
        highlighted_elements_count=highlighted_count,
        element_selectors=element_selectors_list,
    )


if __name__ == "__main__":
    import shutil

    output_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = highlight_interactive_elements_and_screenshot(
        url="https://ph.indeed.com/jobs?q=Machine+Learning+Engineer&l=Remote",
        output_dir="generated",
        headless=False,
        root_selector="body",  # Example: only scan within <main> element
    )
    print(f"Highlighted {result['highlighted_elements_count']} elements")
    print(f"Screenshot: {result['screenshot_path']}")
    print(f"ARIA Snapshot: {result['aria_snapshot_path']}")
    print(f"Coordinates: {result['coordinates_path']}")
    print(f"Elements JSON: {result['interactive_elements_path']}")
