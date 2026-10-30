import base64
import json
import os
import time
from pathlib import Path
from typing import List, TypedDict
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
    events: List[str]


class InteractiveScreenshotResult(TypedDict):
    """Result of taking an interactive element screenshot."""

    url: str
    screenshot_path: str
    screenshot_base64: str
    aria_snapshot_path: str
    aria_snapshot_content: str
    coordinates_path: str
    interactive_elements_path: str
    coordinates: List[ElementCoordinates]
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
) -> InteractiveScreenshotResult:
    """
    Opens a URL, highlights all interactive elements (including divs with JS listeners),
    takes a screenshot, saves an ARIA snapshot, and extracts precise coordinates.
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
    () => {
        const config = {
            color: "%s",
            width: %d
        };

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

        function processElement(el) {
            if (el.dataset.highlighted === 'true') return null;
            if (!isInteractive(el)) return null;
            
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return null;

            // Highlight
            el.style.outline = `${config.width}px solid ${config.color}`;
            el.style.outlineOffset = '2px';
            el.dataset.highlighted = 'true';

            // Generate Selector
            let selector = el.tagName.toLowerCase();
            if (el.id) selector += '#' + el.id;
            else if (el.className) {
                const firstClass = el.className.split(' ')[0];
                if (firstClass) selector += '.' + firstClass;
            }

            const rect = el.getBoundingClientRect();
            return {
                selector: selector,
                x: Math.round(rect.x),
                y: Math.round(rect.y),
                width: Math.round(rect.width),
                height: Math.round(rect.height),
                tag: el.tagName.toLowerCase(),
                text: (el.innerText || el.value || '').trim().substring(0, 50),
                events: getTrackedEvents(el)
            };
        }

        // Process existing elements
        const results = [];
        document.querySelectorAll('*').forEach(el => {
            const data = processElement(el);
            if (data) results.push(data);
        });

        // Observe future elements
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) {
                        processElement(node);
                        node.querySelectorAll('*').forEach(child => processElement(child));
                    }
                });
            });
        });
        observer.observe(document.body, { childList: true, subtree: true });

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

        # Reduced wait duration
        page.wait_for_timeout(500)

        # Execute the detection script
        element_data = page.evaluate(detection_script)

        highlighted_count = len(element_data)
        element_selectors_list = [item["selector"] for item in element_data]
        coordinates_list = element_data

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
        if coordinates_list:
            interactive_elements_filename = (
                f"{safe_host}_{timestamp}_interactive_elements.json"
            )
            interactive_elements_path = os.path.join(
                output_dir, interactive_elements_filename
            )

            # Create a cleaner summary for the interactive elements file
            summary_data = [
                {
                    "selector": el["selector"],
                    "tag": el["tag"],
                    "text": el["text"],
                    "events": el["events"],
                    "position": {"x": el["x"], "y": el["y"]},
                }
                for el in coordinates_list
            ]

            with open(interactive_elements_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)

            logger.success(
                f"Interactive elements summary saved: {interactive_elements_path}"
            )

        browser.close()

    return InteractiveScreenshotResult(
        url=url,
        screenshot_path=screenshot_path,
        screenshot_base64=screenshot_base64,
        aria_snapshot_path=aria_snapshot_path,
        aria_snapshot_content=aria_snapshot_content,
        coordinates_path=coordinates_path,
        interactive_elements_path=interactive_elements_path,
        coordinates=coordinates_list,
        highlighted_elements_count=highlighted_count,
        element_selectors=element_selectors_list,
    )


if __name__ == "__main__":
    result = highlight_interactive_elements_and_screenshot(
        url="https://ph.indeed.com/jobs?q=Machine+Learning+Engineer&l=Remote",
        output_dir="generated",
        headless=False,
    )
    print(f"Highlighted {result['highlighted_elements_count']} elements")
    print(f"Screenshot: {result['screenshot_path']}")
    print(f"ARIA Snapshot: {result['aria_snapshot_path']}")
    print(f"Coordinates: {result['coordinates_path']}")
    print(f"Interactive Elements: {result['interactive_elements_path']}")
