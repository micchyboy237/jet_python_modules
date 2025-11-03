import os
import shutil
from playwright.sync_api import sync_playwright
from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from jet.file.utils import save_file

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

def example_page_evaluate():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE)
        page = browser.new_page()
        page.goto("https://example.com")

        # Evaluate JavaScript in the browser context
        bounding_box = page.evaluate("""
        () => {
            const el = document.querySelector('h1');  // target element
            if (!el) return null;
            const rect = el.getBoundingClientRect();
            return {
                x: rect.x,
                y: rect.y,
                width: rect.width,
                height: rect.height,
                top: rect.top,
                left: rect.left,
                bottom: rect.bottom,
                right: rect.right
            };
        }
        """)

        print("Bounding box:", bounding_box)

        browser.close()

        save_file({
            "bounding_box": bounding_box
        }, f"{OUTPUT_DIR}/page_evaluation.json")

if __name__ == "__main__":
    example_page_evaluate()
