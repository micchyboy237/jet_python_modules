import os
import shutil
from playwright.sync_api import sync_playwright
from pathlib import Path
from typing import Dict

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Utility to launch a page for examples
def _open_example_page(page):
    page.goto("https://example.com")
    page.evaluate("""
        document.body.innerHTML = `
            <div style="height:2000px;background:linear-gradient(white,lightblue);padding:20px;">
                <h1>Playwright Screenshot Demo</h1>
                <p class="dynamic">This is a test paragraph.</p>
                <div class="price" style="background:yellow;padding:4px;">$199.99</div>
                <textarea autofocus style="width:300px;height:50px;">Typing...</textarea>
            </div>
        `;
    """)


# 1️⃣ Basic screenshot (default PNG)
def example_basic_screenshot(page):
    _open_example_page(page)
    page.screenshot(path=f"{OUTPUT_DIR}/basic.png")
    print(f"✅ Saved screenshot at {OUTPUT_DIR}/basic.png")


# 2️⃣ Full page screenshot
def example_full_page_screenshot(page):
    _open_example_page(page)
    page.screenshot(path=f"{OUTPUT_DIR}/full_page.png", full_page=True)
    print(f"✅ Saved full page screenshot at {OUTPUT_DIR}/full_page.png")


# 3️⃣ Screenshot with clipping region
def example_clipped_region(page):
    _open_example_page(page)
    clip_region: Dict[str, float] = {"x": 100, "y": 150, "width": 400, "height": 250}
    page.screenshot(path=f"{OUTPUT_DIR}/clip_region.png", clip=clip_region)
    print(f"✅ Saved clipped region screenshot at {OUTPUT_DIR}/clip_region.png")


# 4️⃣ JPEG screenshot with quality
def example_jpeg_quality(page):
    _open_example_page(page)
    page.screenshot(path=f"{OUTPUT_DIR}/compressed.jpg", type="jpeg", quality=70)
    print(f"✅ Saved compressed JPEG screenshot at {OUTPUT_DIR}/compressed.jpg")


# 5️⃣ Transparent background (omit background)
def example_transparent_background(page):
    _open_example_page(page)
    # Set body background to non-white to show transparency
    page.evaluate("document.body.style.background = 'linear-gradient(red, orange)'")
    page.screenshot(path=f"{OUTPUT_DIR}/transparent.png", omit_background=True)
    print(f"✅ Saved transparent screenshot at {OUTPUT_DIR}/transparent.png")


# 6️⃣ Handling animations (allow vs disabled)
def example_animations_option(page):
    _open_example_page(page)
    # Add animation to demonstrate difference
    page.evaluate("""
        const el = document.createElement('div');
        el.style.cssText = 'width:100px;height:100px;background:blue;animation: move 3s infinite alternate;';
        el.id = 'animated';
        document.body.appendChild(el);
        const style = document.createElement('style');
        style.textContent = '@keyframes move { from { transform: translateX(0);} to { transform: translateX(200px);} }';
        document.head.appendChild(style);
    """)
    # Allow animations
    page.screenshot(path=f"{OUTPUT_DIR}/animation_allow.png", animations="allow")
    print(f"✅ Saved animated screenshot (animations allowed) at {OUTPUT_DIR}/animation_allow.png")
    # Disable animations (freezes frame)
    page.screenshot(path=f"{OUTPUT_DIR}/animation_disabled.png", animations="disabled")
    print(f"✅ Saved animated screenshot (animations disabled) at {OUTPUT_DIR}/animation_disabled.png")


# 7️⃣ Caret visibility option
def example_caret_behavior(page):
    _open_example_page(page)
    # Hide caret (default)
    page.screenshot(path=f"{OUTPUT_DIR}/caret_hide.png", caret="hide")
    print(f"✅ Saved screenshot with caret hidden at {OUTPUT_DIR}/caret_hide.png")
    # Show caret as-is
    page.screenshot(path=f"{OUTPUT_DIR}/caret_initial.png", caret="initial")
    print(f"✅ Saved screenshot with caret initial at {OUTPUT_DIR}/caret_initial.png")


# 8️⃣ CSS vs Device scale
def example_scale_modes(page):
    _open_example_page(page)
    page.screenshot(path=f"{OUTPUT_DIR}/scale_css.png", scale="css")
    print(f"✅ Saved screenshot with CSS scale at {OUTPUT_DIR}/scale_css.png")
    page.screenshot(path=f"{OUTPUT_DIR}/scale_device.png", scale="device")
    print(f"✅ Saved screenshot with device scale at {OUTPUT_DIR}/scale_device.png")


# 9️⃣ Mask sensitive data
def example_mask_elements(page):
    _open_example_page(page)
    price = page.locator(".price")
    page.screenshot(path=f"{OUTPUT_DIR}/masked_default.png", mask=[price])
    print(f"✅ Saved screenshot with default mask at {OUTPUT_DIR}/masked_default.png")
    page.screenshot(path=f"{OUTPUT_DIR}/masked_custom_color.png", mask=[price], mask_color="#000000")
    print(f"✅ Saved screenshot with custom mask color at {OUTPUT_DIR}/masked_custom_color.png")


# 🔟 Custom style injection
def example_style_injection(page):
    _open_example_page(page)
    hide_dynamic_style = """
        .dynamic { visibility: hidden !important; }
        .price { border: 3px solid red; }
    """
    page.screenshot(path=f"{OUTPUT_DIR}/styled.png", style=hide_dynamic_style)
    print(f"✅ Saved screenshot with style injection at {OUTPUT_DIR}/styled.png")


# 11️⃣ Timeout usage
def example_timeout(page):
    _open_example_page(page)
    # Force slow load simulation
    page.wait_for_timeout(1000)
    page.screenshot(path=f"{OUTPUT_DIR}/timeout.png", timeout=5000)
    print(f"✅ Saved screenshot with timeout at {OUTPUT_DIR}/timeout.png")


# 12️⃣ Return bytes (no file)
def example_return_bytes(page):
    _open_example_page(page)
    image_bytes = page.screenshot(type="png")
    Path(f"{OUTPUT_DIR}/in_memory_save.png").write_bytes(image_bytes)
    print(f"✅ Saved in-memory screenshot at {OUTPUT_DIR}/in_memory_save.png")


# Runner (for manual demo)
def run_all_examples():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        examples = [
            example_basic_screenshot,
            example_full_page_screenshot,
            example_clipped_region,
            example_jpeg_quality,
            example_transparent_background,
            example_animations_option,
            example_caret_behavior,
            example_scale_modes,
            example_mask_elements,
            example_style_injection,
            example_timeout,
            example_return_bytes,
        ]

        for fn in examples:
            print(f"Running: {fn.__name__}")
            fn(page)

        browser.close()


if __name__ == "__main__":
    run_all_examples()
