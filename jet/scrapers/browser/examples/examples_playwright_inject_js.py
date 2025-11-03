import os
import shutil
from pathlib import Path
from jet.file.utils import save_file
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from playwright.sync_api import sync_playwright

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

JS_UTILS_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/scrapers/browser/scripts/utils.js"

def example_inject_js():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()

        # 🔹 Inject listener tracker BEFORE navigation (runs before any page JS)
        page.add_init_script("""
        (() => {
          window.__clickableElements = new Set();
          const origAddEventListener = EventTarget.prototype.addEventListener;
          EventTarget.prototype.addEventListener = function(type, listener, options) {
            if (type === 'click' && this instanceof Element) {
              window.__clickableElements.add(this);
            }
            return origAddEventListener.call(this, type, listener, options);
          };
        })();
        """)

        # Now navigate normally
        page.goto("https://example.com")

        # Inject our JS utilities (after load)
        page.add_script_tag(path=JS_UTILS_PATH)

        print("✅ Injected utils.js and click-tracker")

        # ---- Demonstrate all utils ----
        message = page.evaluate("Utils.myInjectedFunction('Jet')")
        print("myInjectedFunction:", message)

        bbox = page.evaluate("Utils.getBoundingBox('h1')")
        print("getBoundingBox('h1'):", bbox)

        scrolled = page.evaluate("Utils.scrollIntoView('h1')")
        print("scrollIntoView('h1'):", scrolled)

        leaf_texts = page.evaluate("Utils.getLeafTexts('body')")
        print("getLeafTexts('body'):", leaf_texts[:5], "..." if len(leaf_texts) > 5 else "")

        clickables = page.evaluate("Utils.getClickableElements()")
        print(f"getClickableElements(): Found {len(clickables)} elements")
        for c in clickables[:3]:
            print(" -", c)
        save_file(clickables, f"{OUTPUT_DIR}/clickables.json")

        # ---- Collect elements that had JS click listeners attached ----
        js_clickables = page.evaluate("""
        Array.from(window.__clickableElements).map(el => ({
            tag: el.tagName.toLowerCase(),
            text: el.innerText?.trim().slice(0, 100) || '',
            hasHref: !!el.getAttribute('href')
        }))
        """)
        print(f"Detected {len(js_clickables)} elements with JS click listeners")
        for el in js_clickables[:3]:
            print(" -", el)
        save_file(js_clickables, f"{OUTPUT_DIR}/js_clickables.json")

        # ---- Capture screenshot for reference ----
        element = page.query_selector("h1")
        if element:
            screenshot_path = Path(os.path.join(OUTPUT_DIR, "injected_h1_screenshot.png")).resolve()
            element.screenshot(path=str(screenshot_path))
            print(f"✅ Screenshot saved at {screenshot_path}")
        else:
            print("❌ Element for screenshot not found")

        browser.close()

if __name__ == "__main__":
    example_inject_js()
