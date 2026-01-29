import base64
from pathlib import Path
import random
import shutil
import time
import helium
from typing import List, Literal, Optional
import os
from datetime import datetime

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support.ui import Select
from seleniumbase import Driver

from jet.utils.text import format_sub_source_dir

LinkMode = Literal["full", "fast", "smart"]
TextMode = Literal["full", "fast", "smart"]
ListItemMode = Literal["full", "fast", "smart"]


def init_browser(headless: bool = True) -> "Driver":
    """
    Initialize an anti-detection browser instance using SeleniumBase UC mode.
    """
    # Optional: rotate user-agent per script run (helps long-running agents)
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",
    ]

    selected_ua = random.choice(user_agents)

    driver = Driver(
        browser="chrome",
        uc=True,
        headless=headless,
        agent=selected_ua,
        window_size="1000,1350",
        window_position="500,0",
        d_p_r=1.0,
        chromium_arg="--disable-pdf-viewer",
    )

    helium.set_driver(driver)

    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {
            "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
            """
        },
    )

    # Increase timeout for slow / strict sites
    driver.set_page_load_timeout(45)

    return driver


class DemoHeliumActions:
    """
    Demonstrates the most common Helium high-level actions in a clean, reusable class.
    Each method shows one primary action + typical usage pattern.
    """

    def __init__(
        self, url: str, headless: bool = True, output_dir: Optional[str | Path] = None
    ):
        self.url = url

        self.driver: WebDriver = init_browser(headless=headless)
        self.output_dir = (
            str(output_dir)
            if output_dir
            else str(Path(__file__).parent / "generated" / Path(__file__).stem)
        )
        self.sample_file_for_upload = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/__sample.txt"

        shutil.rmtree(self.output_dir, ignore_errors=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.demo_go_to(url)

    def close(self):
        """Clean up browser"""
        helium.kill_browser()

    def get_driver(self) -> WebDriver:
        """Returns the current Selenium WebDriver instance"""
        if self.driver is None:
            raise RuntimeError("Browser driver not initialized.")
        return self.driver

    def get_current_url(self) -> str:
        """Returns the current page URL"""
        if self.driver is None:
            raise RuntimeError("Browser driver not initialized.")
        url = self.driver.current_url
        print(f"Current URL: {url}")
        return url

    def get_page_source(self) -> str:
        """Returns the full current HTML page source"""
        if self.driver is None:
            raise RuntimeError("No browser driver is currently set.")
        source = self.driver.page_source
        preview_len = 600
        preview = source[:preview_len] + ("..." if len(source) > preview_len else "")
        print(f"Page source length: {len(source)} | Preview:\n{preview}")
        return source

    def print_browser_state(self):
        """Quick helper to log both URL and source preview"""
        self.get_current_url()
        self.get_page_source()

    def demo_go_to(self, url: str):
        """Action: go_to(url)"""
        helium.go_to(url)
        print("→ Navigated to trytestingthis.netlify.app demo page")

    def demo_click(self):
        """Action: click(element) — most common way to interact"""
        helium.click("Your Sample Alert Button!")
        print("→ Clicked 'Your Sample Alert Button!' (triggers JS alert)")

        # Handle the JavaScript alert
        helium.Alert().accept()
        print("→ Accepted (OK'd) the alert popup")

    def demo_write(self):
        """Action: write(text, into=field)"""
        helium.write("LeBron", into="First name:")
        print("→ Typed 'LeBron' into First name field")

        helium.write("James", into="Last name:")
        print("→ Typed 'James' into last name field")

    def demo_press_keys(self):
        """Action: press(*keys) — keyboard input"""
        helium.press(helium.ENTER)
        print("→ Pressed ENTER (should submit login form)")

    def demo_select_dropdown(self):
        """Action: select(option, element) — demonstrates single & multi dropdowns"""
        print("→ Attempting single-select dropdown...")
        try:
            # Most reliable: target by ID and unwrap to WebElement
            single_select_elem = helium.S("#option").web_element
            Select(single_select_elem).select_by_visible_text("Option 1")
            print("→ Successfully selected 'Option 1' from <select id='option'>")
        except Exception as e:
            print(f"→ Single-select failed: {e}")
            # Fallback attempt using label + relative
            try:
                helium.select(
                    "Option 1",
                    helium.S("select", below="Lets you select only one option"),
                )
                print("→ Fallback success using relative locator")
            except Exception as fallback_e:
                print(f"→ All single-select attempts failed: {fallback_e}")

        print(
            "\n→ Attempting multi-select (note: page has <select multiple id='owc'>)..."
        )
        try:
            multi_select_elem = helium.S("#owc").web_element
            Select(multi_select_elem).select_by_visible_text("Option 2")
            Select(multi_select_elem).select_by_visible_text(
                "Option 3"
            )  # multiple calls to select more
            print("→ Selected 'Option 2' and 'Option 3' in multi-select")
        except Exception as e:
            print(
                f"→ Multi-select failed: {e} (check if id='owc' is correct and multiple attribute present)"
            )

    def demo_wait_until_and_exists(self):
        """Demonstrates explicit waiting and existence checks"""
        helium.Config.implicit_wait_secs = 10
        print("→ Waiting for alert button...")

        # Modern / clean style:
        alert_button = helium.S("button[onclick='alertfunction()']")
        # or: alert_button = helium.Button(helium.S("[onclick='alertfunction()']"))

        print(f"Immediate check: alert button exists? {alert_button.exists()}")

        try:
            helium.wait_until(alert_button.exists, timeout_secs=20)
            print("→ Success: waited until alert button exists")
            # Optional: interact
            # alert_button.click()
            # helium.Alert().accept()
        except TimeoutException:
            print("→ Timeout: alert button still not found – debug info:")
            all_buttons = helium.find_all(helium.S("button"))
            print(f"Found {len(all_buttons)} <button> elements:")
            for btn in all_buttons:
                txt = btn.web_element.text.strip()
                onclick = btn.web_element.get_attribute("onclick") or "N/A"
                print(f" • Text='{txt}' | onclick='{onclick}'")

        print("→ Waiting for sample table legend...")
        table_legend_text = helium.Text("This is your Sample Table:")
        helium.wait_until(table_legend_text.exists, timeout_secs=10)
        print("→ Success: waited until sample table legend exists")

    def demo_s_selector_and_relative(self):
        """Shows S() selector + relative positioning"""
        uname_field = helium.S("#uname")
        if uname_field.exists():
            print("→ Found username field via S('#uname')")

        # Example relative locator (adjust based on actual layout if needed)
        password_label = helium.Text(to_right_of="Password:")
        if password_label.exists():
            print(f"→ Found relative text near password: {password_label.value}")

    def demo_find_all_elements(self):
        """Extract multiple matching elements with find_all()"""
        # All table cells (basic)
        table_cells = helium.find_all(helium.S("table tr td"))
        print(f"→ Found {len(table_cells)} table cells in sample table")

        # Example using relative locator with REAL existing text
        print("\n→ Trying to find cells below an actual header ('Age')...")
        try:
            age_header = "Age"  # exists on the page
            age_cells = helium.find_all(helium.S("td", below=age_header))
            ages = [
                cell.web_element.text.strip()
                for cell in age_cells
                if cell.web_element.text.strip()
            ]
            print(f"→ Found {len(ages)} values below 'Age' header:")
            for age in ages:
                print(f" • {age}")
        except LookupError:
            print("→ LookupError: reference text not found – check page content")

        # Fallback: column by index (very reliable)
        print("\n→ Extracting 'Occupation' column by cell index (safer)...")
        occupations = []
        for i, cell in enumerate(table_cells):
            if i % 5 == 4:  # 5 columns, occupation is last
                text = cell.web_element.text.strip()
                if text:
                    occupations.append(text)
        print("→ Occupations:")
        for occ in occupations:
            print(f" • {occ}")

    def demo_scroll(self):
        """Action: scroll_down() / scroll_up()"""
        helium.scroll_down(400)
        print("→ Scrolled down 400px on demo page")
        helium.scroll_up()
        print("→ Scrolled back to top")

    # -------- New Demo Methods --------

    def demo_link_element(self):
        """Demonstrates Link element locator"""
        try:
            helium.click(
                helium.Link("Click Here")
            )  # replace with actual link text if exists
            print("→ Clicked Link('Click Here')")
        except Exception:
            print("→ No suitable Link found for demo - check page for <a> tags")

    def demo_image_element(self):
        """Demonstrates Image locator (alt text or src)"""
        img = helium.Image(alt="")  # or helium.Image(src_contains="pizza")
        if img.exists():
            src = img.web_element.get_attribute("src")
            print(f"→ Found image with src: {src}")
        else:
            print("→ No Image with matching alt/src found - adjust locator")

    def demo_double_click(self):
        """Action: doubleclick(element) – triggers JS to update text below button"""
        button = helium.Button("Double-click me")
        if button.exists():
            helium.doubleclick(button)
            print("→ Double-clicked 'Double-click me' button")
            print(
                "→ Expected result: text 'Your Sample Double Click worked!' should appear below"
            )
        else:
            print("→ Double-click button not found")

    def demo_drag_and_drop(self):
        """Action: drag(source, to=target) — uses S() with CSS/ID for reliability"""
        # Best: Use S() with ID selector (most reliable, as Image doesn't support src directly)
        source = helium.S("#drag1")
        # Alternative: CSS for src exact match (if ID not available)
        # source = helium.S('img[src="pizza.gif"]')
        # For partial src: helium.S('img[src*="pizza.gif"]')  # CSS contains operator

        target = helium.S("#div1")

        if source.exists() and target.exists():
            helium.drag(source, to=target)
            print("→ Dragged pizza image to drop zone (id=div1)")
            # Optional: Add wait to observe or verify drop (e.g., check if image moved via JS)
            # helium.wait_until(lambda: "dropped" in helium.S("#div1 img").web_element.get_attribute("src") or similar)
        else:
            print("→ Drag source or target not found")
            print(f"  • Source exists? {source.exists()}")
            print(f"  • Target exists? {target.exists()}")

    def demo_file_upload(self):
        """Demonstrates file upload attempt on the main demo page"""
        if not os.path.exists(self.sample_file_for_upload):
            print(f"→ Error: Sample file missing at {self.sample_file_for_upload}")
            return

        try:
            # Try common file input locators
            file_input = self.driver.find_element("css selector", "input[type='file']")
            file_input.send_keys(self.sample_file_for_upload)
            print("→ Attached file via send_keys to first <input type='file'> found")
        except Exception as e:
            print(f"→ No file input found or upload failed: {e}")
            # Debug: List all inputs to confirm
            all_inputs = self.driver.find_elements("tag name", "input")
            print(f"→ Page has {len(all_inputs)} <input> elements:")
            for inp in all_inputs:
                attrs = f"id={inp.get_attribute('id') or 'N/A'}, type={inp.get_attribute('type') or 'N/A'}, name={inp.get_attribute('name') or 'N/A'}"
                print(
                    f" • {attrs} | outerHTML preview: {inp.get_attribute('outerHTML')[:120]}..."
                )
            print(
                "→ If no type='file' input exists, this demo page simply doesn't support file upload."
            )

    def demo_login_form(self):
        """Fills and submits the sample login form"""
        helium.write("test", into="Username:test")
        helium.write("test", into="Password:test")
        helium.press(helium.ENTER)
        print("→ Submitted login with test/test – expect redirect or success")

    def demo_single_dropdown(self):
        """Selects from single-select dropdown"""
        helium.select("Option 2", "Choose an option:")
        print("→ Selected 'Option 2' from single-select dropdown")

    def demo_checkbox(self):
        """Clicks checkboxes using labels – adapted for the demo page (which has none)"""
        print(
            "→ Attempting checkbox demo on main page (note: page uses dropdowns, not checkboxes)..."
        )

        # Try your original (will likely fail, but for demo)
        try:
            helium.click("Option 1")  # no leading space – try without
            print("→ Clicked 'Option 1' (if it existed as checkbox)")
        except LookupError:
            print(
                "→ No element with text 'Option 1' found (expected – page has no checkboxes)"
            )

        try:
            helium.click(" Option 1")  # your original with space
            print("→ Clicked ' Option 1'")
        except LookupError:
            print("→ No element with exact text ' Option 1' (leading space mismatch)")

        # Better: Show that multi-select exists instead
        print(
            "\n→ Page has multi-select dropdown instead of checkboxes. Demonstrating selection..."
        )
        try:
            multi_select = helium.S("#owc")  # from earlier multi-select ID
            if multi_select.exists():
                # Use raw Selenium for multi-select (Helium select() can be finicky)
                select_elem = self.driver.find_element("id", "owc")
                from selenium.webdriver.support.select import Select

                sel = Select(select_elem)
                sel.select_by_visible_text("Option 2")
                sel.select_by_visible_text("Option 3")
                print(
                    "→ Selected 'Option 2' and 'Option 3' in multi-select (alternative to checkboxes)"
                )
            else:
                print("→ Multi-select #owc not found")
        except Exception as e:
            print(f"→ Multi-select demo failed: {e}")

    def demo_radio_button(self):
        """Real working radio button demo on a page that actually has them"""
        print("\n=== Real Working Radio Button Demo ===\n")

        try:
            helium.wait_until(helium.S("input[type='radio']").exists, timeout_secs=10)
            print("→ Radio buttons found on page")
        except TimeoutException:
            print("→ No radio buttons detected – page may have changed")
            return

        # Different reliable ways to select radio buttons
        strategies = [
            ("By visible label text", lambda: helium.click("Male")),
            (
                "By CSS value attribute",
                lambda: helium.click(helium.S("input[value='female']")),
            ),
            ("By ID (if known)", lambda: helium.click(helium.S("#other"))),
            ("First unselected radio", self._click_first_unselected_radio),
        ]

        for name, action in strategies:
            try:
                action()
                print(f"  ✓ {name} succeeded")
                # Brief pause to see the selection
                time.sleep(1.5)
            except Exception as e:
                print(f"  ✗ {name} failed → {type(e).__name__}")

        # Show which one is selected
        selected_value = self._get_selected_radio_value()
        print(f"\n→ Final selected radio value: {selected_value or 'None'}")

    def _click_first_unselected_radio(self):
        for radio in helium.find_all(helium.S("input[type='radio']")):
            if not radio.web_element.is_selected():
                helium.click(radio)
                print("    → Clicked first unselected radio")
                return
        raise LookupError("No unselected radio found")

    def _get_selected_radio_value(self):
        for radio in helium.find_all(helium.S("input[type='radio']")):
            if radio.web_element.is_selected():
                return radio.web_element.get_attribute("value") or "?"
        return None

    def demo_point_click(self):
        """Action: click(Point(x, y)) - coordinate click"""
        # Force scroll to top using JS (most reliable)
        self.driver.execute_script("window.scrollTo(0, 0);")
        print("→ Scrolled to top via JS for absolute coordinate reliability")

        point = helium.Point(
            x=100, y=100
        )  # Very safe: near top-left, usually body area
        try:
            helium.click(point)
            print(f"→ Successfully clicked at absolute coordinates {point}")
        except Exception as e:
            print(f"→ Click at {point} failed: {type(e).__name__} - {str(e)}")
            # Fallback: raw ActionChains (relative to current mouse, but can reset)
            from selenium.webdriver.common.action_chains import ActionChains

            actions = ActionChains(self.driver)
            actions.move_by_offset(100, 100).click().perform()
            print("→ Fallback raw ActionChains click to (100,100) succeeded")

        # Optional debug: current scroll position
        scroll_pos = self.driver.execute_script("return window.pageYOffset;")
        print(f"→ Current scroll Y position after attempt: {scroll_pos}")

    def demo_read_values(self):
        """Reading values from elements"""
        # Read text content
        heading = helium.Text("Try Testing This")
        if heading.exists():
            print(f"→ Page title text: {heading.value}")

        # Read input value (after typing earlier)
        uname_field = helium.TextField("Username:test")
        if uname_field.exists():
            print(f"→ Username field value: {uname_field.value}")

    def demo_take_screenshot(self):
        """Take a screenshot of a specific element (fallback to viewport)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"helium_element_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        # Example: stable element that exists on the page
        selector = "table"  # change freely (e.g. '#uname', '.container', 'form')

        try:
            element = self.driver.find_element("css selector", selector)
            element.screenshot(filepath)
            print(f"→ Element screenshot saved ({selector}): {filepath}")
        except Exception as e:
            print(f"→ Element screenshot failed ({selector}): {e}")
            self.driver.save_screenshot(filepath)
            print(f"→ Fallback viewport screenshot saved: {filepath}")

        print(f"   Full path: {os.path.abspath(filepath)}")

    def demo_take_full_page_screenshot(self):
        """Capture full scrollable page by resizing window to content size"""
        print("\n=== Full-Page Screenshot (Resize Method - Reliable & No CDP) ===\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_page_resize_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Save original window size
            original_size = self.driver.get_window_size()

            # Get full content dimensions (most accurate combo)
            width_script = """
                return Math.max(
                    document.body.scrollWidth,
                    document.body.offsetWidth,
                    document.documentElement.clientWidth,
                    document.documentElement.scrollWidth,
                    document.documentElement.offsetWidth
                );
            """
            height_script = """
                return Math.max(
                    document.body.scrollHeight,
                    document.body.offsetHeight,
                    document.documentElement.clientHeight,
                    document.documentElement.scrollHeight,
                    document.documentElement.offsetHeight
                );
            """

            width = self.driver.execute_script(width_script)
            height = self.driver.execute_script(height_script)

            print(f"→ Detected full page size: {width}px width × {height}px height")

            # Reset scroll to avoid partial renders
            self.driver.execute_script("window.scrollTo(0, 0);")

            # Resize window (Chrome needs buffer)
            self.driver.set_window_size(width + 120, height + 300)

            # Slightly longer delay for resize to settle
            time.sleep(1.2)

            self.driver.save_screenshot(filepath)
            print(f"→ Full-page screenshot saved: {filepath}")
            print(f"  Full path: {os.path.abspath(filepath)}")

            # Restore original window size
            self.driver.set_window_size(original_size["width"], original_size["height"])

        except Exception as e:
            print(f"→ Resize full-page capture failed: {e}")
            # Fallback: plain viewport screenshot
            self.driver.save_screenshot(filepath)
            print(f"→ Saved viewport-only screenshot instead: {filepath}")

    def demo_take_full_page_overflow_screenshot(self):
        """True full-page screenshot using Chrome DevTools Protocol"""
        print("\n=== Full-Page Overflow Screenshot (CDP - TRUE FULL PAGE) ===\n")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_page_overflow_cdp_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)

        try:
            # Get layout metrics
            metrics = self.driver.execute_cdp_cmd("Page.getLayoutMetrics", {})

            content_size = metrics["contentSize"]
            width = content_size["width"]
            height = content_size["height"]

            print(f"→ Layout viewport size: {width} × {height}")

            screenshot = self.driver.execute_cdp_cmd(
                "Page.captureScreenshot",
                {
                    "format": "png",
                    "captureBeyondViewport": True,
                    "clip": {
                        "x": 0,
                        "y": 0,
                        "width": width,
                        "height": height,
                        "scale": 1,
                    },
                },
            )

            with open(filepath, "wb") as f:
                f.write(base64.b64decode(screenshot["data"]))

            print(f"→ TRUE full-page screenshot saved: {filepath}")

        except Exception as e:
            print(f"→ CDP full-page capture failed: {e}")
            self.driver.save_screenshot(filepath)
            print("→ Fallback viewport screenshot saved")

    from selenium.webdriver.common.by import By

    LinkMode = Literal["full", "fast", "smart"]

    def demo_read_links(
        self,
        mode: LinkMode = "full",
        max_links: int = 0,  # 0 = no limit
        only_textual: bool = True,  # skip <a> without text (nav icons etc.)
    ):
        """Extracts links from the page using different strategies/modes.

        Modes:
          - 'full'   : uses helium.Link('') → slowest, most accurate visibility/text
          - 'fast'   : raw Selenium <a> tags → much faster, includes hidden
          - 'smart'  : JavaScript-based → fastest, focuses on content links
        """
        print(f"\n=== Reading links (mode={mode!r}, max={max_links or 'all'}) ===\n")

        links_data: list[dict[str, str]] = []

        start = time.perf_counter()

        try:
            if mode == "full":
                all_links = helium.find_all(helium.Link(""))
                print(f"→ Helium Link('') found {len(all_links)} elements")

                for i, link in enumerate(all_links, 1):
                    if max_links and i > max_links:
                        break
                    text = (link.web_element.text or "").strip()
                    href = link.href or link.web_element.get_attribute("href") or ""
                    if only_textual and not text:
                        continue
                    links_data.append({"text": text, "href": href})
                    print(f"  {i:3d}. {text:.<40} → {href}")

            elif mode == "fast":
                raw_as = self.driver.find_elements(By.TAG_NAME, "a")
                print(f"→ Raw <a> tags found {len(raw_as)} elements")

                for i, a in enumerate(raw_as, 1):
                    if max_links and i > max_links:
                        break
                    text = (a.text or "").strip()
                    href = a.get_attribute("href") or ""
                    if only_textual and not text:
                        continue
                    links_data.append({"text": text, "href": href})
                    print(f"  {i:3d}. {text:.<40} → {href}")

            elif mode == "smart":
                js_code = r"""
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({
                        text: (a.textContent || '').trim(),
                        href: a.href || ''
                    }))
                    .filter(item => item.text.length > 0 && item.href);
                """
                if max_links:
                    js_code = js_code.replace(
                        ".filter(item => item.text.length > 0 && item.href);",
                        f".slice(0, {max_links}).filter(item => item.text.length > 0 && item.href);",
                    )

                result = self.driver.execute_script(js_code)
                print(f"→ JavaScript collected {len(result)} content links")

                for i, item in enumerate(result, 1):
                    text = item["text"]
                    href = item["href"]
                    links_data.append({"text": text, "href": href})
                    print(f"  {i:3d}. {text:.<40} → {href}")

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            print(f"→ Failed in mode {mode!r}: {type(exc).__name__} - {exc}")

        duration = time.perf_counter() - start
        print(
            f"\n→ Completed in {duration:.2f} seconds | collected {len(links_data)} links"
        )

        return links_data

    def demo_read_buttons(self):
        """Demonstrates extracting button texts using Button() locator"""
        print("\n=== Demo: Reading all visible button labels ===\n")
        try:
            all_buttons: List[helium.Button] = helium.find_all(helium.Button(""))
            print(f"→ Found {len(all_buttons)} Button elements via Button('')")

            shown = 0
            for i, btn in enumerate(all_buttons, 1):
                label = (
                    btn.web_element.text.strip()
                    or btn.web_element.get_attribute("value")
                    or "[No text]"
                )
                enabled = btn.is_enabled()
                print(f"  {i}. Button text: {label:.<30} (enabled: {enabled})")
                shown += 1

        except Exception as e:
            print(f"→ Error reading buttons via Button(): {type(e).__name__} - {e}")

        if len(all_buttons) == 0:
            # Fallback: raw <button> + <input type=button/submit/reset>
            print("  → No buttons via Button('') → showing raw candidates")
            print("\n→ Fallback: raw <button> and input[type=button/submit]")
            try:
                candidates = (
                    helium.find_all(helium.S("button"))
                    + helium.find_all(helium.S("input[type='button']"))
                    + helium.find_all(helium.S("input[type='submit']"))
                    + helium.find_all(
                        helium.S("input[type='reset']")
                    )  # sometimes useful
                )
                for i, el in enumerate(candidates, 1):
                    tag = el.web_element.tag_name
                    txt = (
                        el.web_element.text
                        or el.web_element.get_attribute("value")
                        or ""
                    ).strip()
                    print(f"    {i}. <{tag}> → '{txt}'")
            except Exception as e:
                print(f"  Fallback failed: {e}")
        else:
            print(
                "\n→ High-level Button locator succeeded → skipping detailed fallback"
            )

    def demo_read_texts(
        self,
        mode: TextMode = "full",
        max_texts: int = 0,  # 0 = no limit
        min_length: int = 4,  # skip very short fragments
        skip_whitespace_only: bool = True,
    ):
        """Demonstrates different ways of extracting text content from the page.

        Modes:
          - 'full'   : uses helium.find_all(Text('')) → includes coordinates, slowest
          - 'fast'   : raw Selenium elements with text nodes → faster, no coordinates
          - 'smart'  : JavaScript → focuses on meaningful visible text, fastest
        """
        print(
            f"\n=== Demo: Reading texts (mode={mode!r}, max={max_texts or 'all'}, min_len={min_length}) ===\n"
        )

        from typing import Any, Dict

        collected: List[Dict[str, Any]] = []
        start = time.perf_counter()

        try:
            if mode == "full":
                all_texts = helium.find_all(helium.Text(""))
                print(f"→ Helium Text('') found {len(all_texts)} text elements")

                for i, t in enumerate(all_texts, 1):
                    if max_texts and i > max_texts:
                        break

                    val = (t.value or "").strip()
                    if (
                        not val
                        or (skip_whitespace_only and val.isspace())
                        or len(val) < min_length
                    ):
                        continue

                    collected.append(
                        {
                            "text": val,
                            "x": t.x,
                            "y": t.y,
                        }
                    )

                    preview = val[:75] + ("..." if len(val) > 75 else "")
                    pos = (
                        f"(x≈{t.x:.0f}, y≈{t.y:.0f})"
                        if t.x is not None and t.y is not None
                        else ""
                    )
                    print(f"  {len(collected):3d}. {preview!r} {pos}")

            elif mode == "fast":
                # Collect elements that have direct text content
                elements = self.driver.find_elements(
                    By.XPATH, "//*[normalize-space(text()) != '']"
                )
                print(f"→ Raw elements with text found {len(elements)}")

                for i, el in enumerate(elements, 1):
                    if max_texts and i > max_texts:
                        break

                    text = (el.text or "").strip()
                    if not text or len(text) < min_length:
                        continue

                    collected.append({"text": text})
                    preview = text[:75] + ("..." if len(text) > 75 else "")
                    print(f"  {len(collected):3d}. {preview!r}")

            elif mode == "smart":
                js_code = r"""
                function getVisibleText() {
                    const walker = document.createTreeWalker(
                        document.body,
                        NodeFilter.SHOW_TEXT,
                        {
                            acceptNode: node => {
                                const parent = node.parentElement;
                                if (!parent) return NodeFilter.FILTER_REJECT;
                                const style = window.getComputedStyle(parent);
                                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
                                    return NodeFilter.FILTER_REJECT;
                                }
                                return NodeFilter.FILTER_ACCEPT;
                            }
                        }
                    );

                    const texts = [];
                    let node;
                    while (node = walker.nextNode()) {
                        const text = node.nodeValue.trim();
                        if (text.length >= arguments[0]) {
                            texts.push(text);
                        }
                    }
                    return texts.slice(0, arguments[1] || texts.length);
                }
                return getVisibleText(arguments[0], arguments[1]);
                """

                result = self.driver.execute_script(
                    js_code, min_length, max_texts if max_texts > 0 else 999999
                )

                print(f"→ JavaScript visible text nodes collected {len(result)} items")

                for i, text in enumerate(result, 1):
                    collected.append({"text": text})
                    preview = text[:75] + ("..." if len(text) > 75 else "")
                    print(f"  {i:3d}. {preview!r}")

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            print(f"→ Failed in mode {mode!r}: {type(exc).__name__} - {exc}")

        duration = time.perf_counter() - start
        print(
            f"\n→ Completed in {duration:.2f} seconds | collected {len(collected)} texts"
        )

        # # Keep the targeted examples (they are fast and useful)
        # print("\nTargeted examples (independent of mode):")
        # examples = [
        #     ("Sample Table header", helium.Text("This is your Sample Table:")),
        #     (
        #         "Below Monika, right of Occupation",
        #         helium.Text(below="Monika", to_right_of="Occupation"),
        #     ),
        #     ("The cat sentence", helium.Text("The cat was playing in the garden.")),
        # ]
        # for desc, t in examples:
        #     try:
        #         helium.wait_until(t.exists, timeout_secs=3)
        #         print(f"  → {desc}: {t.value.strip()!r}")
        #     except TimeoutException:
        #         print(f"  → {desc} not found (timeout)")

        return collected

    def demo_read_list_items(
        self,
        mode: ListItemMode = "full",
        max_items: int = 0,  # 0 = no limit
        min_length: int = 3,  # skip very short / empty items
    ):
        """
        Demonstrates different ways of extracting <li> list items from the page.

        Modes:
          - 'full'   : uses helium.find_all(ListItem('')) → Helium semantics, slowest
          - 'fast'   : raw Selenium <li> elements → fast, no wrapping
          - 'smart'  : JavaScript → collects visible meaningful <li> text, fastest
        """
        print(
            f"\n=== Demo: Reading list items (mode={mode!r}, max={max_items or 'all'}, min_len={min_length}) ===\n"
        )

        collected: List[Dict[str, str]] = []
        start = time.perf_counter()

        try:
            if mode == "full":
                all_items: List[helium.ListItem] = helium.find_all(helium.ListItem(""))
                print(f"→ Helium ListItem('') found {len(all_items)} items")

                for i, item in enumerate(all_items, 1):
                    if max_items and i > max_items:
                        break
                    text = (item.web_element.text or "").strip()
                    if len(text) < min_length:
                        continue
                    collected.append({"text": text})
                    print(f"  {len(collected):3d}. {text}")

            elif mode == "fast":
                raw_li = self.driver.find_elements(By.TAG_NAME, "li")
                print(f"→ Raw <li> elements found {len(raw_li)}")

                for i, li in enumerate(raw_li, 1):
                    if max_items and i > max_items:
                        break
                    text = (li.text or "").strip()
                    if len(text) < min_length:
                        continue
                    collected.append({"text": text})
                    print(f"  {len(collected):3d}. {text}")

            elif mode == "smart":
                js_code = r"""
                return Array.from(document.querySelectorAll('li'))
                    .map(li => li.textContent.trim())
                    .filter(text => text.length >= arguments[0])
                    .slice(0, arguments[1] || Infinity);
                """
                result = self.driver.execute_script(
                    js_code, min_length, max_items if max_items > 0 else 999999
                )

                print(f"→ JavaScript collected {len(result)} visible list items")

                for i, text in enumerate(result, 1):
                    collected.append({"text": text})
                    print(f"  {i:3d}. {text}")

            else:
                raise ValueError(f"Unknown mode: {mode}")

        except Exception as exc:
            print(f"→ Failed in mode {mode!r}: {type(exc).__name__} - {exc}")

        duration = time.perf_counter() - start
        print(
            f"\n→ Completed in {duration:.2f} seconds | collected {len(collected)} items"
        )

        if not collected:
            print("→ No list items found (or all filtered out)")

            # Keep useful debug: raw count
            raw_count = len(self.driver.find_elements(By.TAG_NAME, "li"))
            print(f"  Raw <li> count on page: {raw_count}")

        print("\nTip: To test properly, navigate to a page with lists, e.g.:")
        print(
            "  helium.go_to('https://en.wikipedia.org/wiki/Python_(programming_language)')"
        )
        print("  # or any site with <ul>, <ol>, FAQs, menus, etc.")

        return collected

    def demo_highlight_element(self):
        """Shows highlight(element) – draws red rectangle (good for visual debug)"""
        print("\n=== Demo: highlight() ===\n")
        try:
            # Example 1: highlight by text (looks for Button, TextField, etc.)
            print("→ Highlighting button with text 'Your Sample Alert Button!'")
            helium.highlight("Your Sample Alert Button!")
            time.sleep(2.5)  # Give time to see the red border

            # Example 2: highlight via S() selector
            print("→ Highlighting username field")
            helium.highlight(helium.S("#uname"))
            time.sleep(2)
            print("→ Highlight demo finished")
        except Exception as e:
            print(f"→ highlight() failed: {e}")
            print("  (Note: highlight may not work in headless mode or certain pages)")

    def demo_attach_file(self):
        """Demonstrates attach_file(file_path, to=...)"""
        print("\n=== Demo: attach_file() – file upload ===\n")
        # You need a real file path that exists
        if not os.path.exists(self.sample_file_for_upload):
            print(f"→ Sample file not found at {self.sample_file_for_upload}")
            print("  → Create a dummy file or adjust path to test this demo.")
            return
        try:
            # Most common pattern: attach to label text next to file input
            print(f"→ Trying: attach_file({self.sample_file_for_upload!r}, to=...)")
            helium.attach_file(
                self.sample_file_for_upload, to="Select a file"
            )  # adjust label if needed
            print("→ attach_file succeeded (no exception)")
            # Optional: check if file name appears in UI after attach
            time.sleep(1.5)
            file_name_shown = helium.Text(os.path.basename(self.sample_file_for_upload))
            if file_name_shown.exists():
                print(f"→ File name '{file_name_shown.value}' appeared → looks good")
            else:
                print(
                    "→ File name not visible in UI (but attach may still have worked)"
                )
        except LookupError:
            print(
                "→ Could not find file input with that label → trying first file input"
            )
            try:
                helium.attach_file(
                    self.sample_file_for_upload
                )  # omit 'to=' → uses first <input type=file>
                print("→ attach_file succeeded using first file input on page")
            except Exception as e:
                print(f"→ attach_file still failed: {e}")
        except Exception as e:
            print(f"→ attach_file error: {type(e).__name__} - {e}")

    def demo_press_keys_advanced(self):
        """More complete & safe demo of press(...) with special keys & combos.
        Avoids pressing ENTER in submitting forms to prevent navigation.
        Uses the textarea (non-submitting) where possible.
        """
        print(
            "\n=== Demo: press(key) – special keys & combinations (safe version) ===\n"
        )
        try:
            # Step 1: Find a safe, focusable element that doesn't submit on ENTER
            # The textarea under "Layout two" is ideal (adds newline instead of submit)
            print(
                "→ Locating the description textarea (should not navigate on ENTER)..."
            )
            textarea = helium.S("textarea")  # there's only one on the page
            if not textarea.exists():
                print(
                    "→ Textarea not found → falling back to 'First name:' field (but won't press ENTER)"
                )
                textarea = helium.S("#fname")  # or "First name:"

            # Focus it safely
            helium.click(textarea)
            print("→ Focused the textarea/input field")
            time.sleep(0.8)

            # Safe actions that won't cause navigation
            print("→ Typing some text first...")
            helium.write("Hello")

            print("→ Pressing BACK_SPACE ×3 (delete last 3 chars)")
            helium.press(helium.BACK_SPACE * 3)
            time.sleep(0.6)

            print("→ Pressing ARROW_LEFT ×2 (move cursor left)")
            helium.press(helium.ARROW_LEFT * 2)
            time.sleep(0.6)

            print("→ Typing more text at new cursor position")
            helium.write(" [edited]")

            print(
                "→ Pressing ENTER — should only add newline in textarea (no navigation)"
            )
            helium.press(helium.ENTER)
            time.sleep(1.0)

            print("→ Pressing CONTROL + A (select all text)")
            helium.press(helium.CONTROL + "a")
            time.sleep(0.8)

            print("→ Pressing CONTROL + C (copy)")
            helium.press(helium.CONTROL + "c")
            time.sleep(0.6)

            print(
                "→ Pressing ESC (usually closes popups/dialogs if any — harmless here)"
            )
            helium.press(helium.ESCAPE)
            time.sleep(0.6)

            # Optional: show something happened
            current_text = textarea.web_element.get_attribute("value") or ""
            preview = current_text[:60] + ("..." if len(current_text) > 60 else "")
            print(f"→ Current field content preview: {preview!r}")

            print("→ All key presses completed without navigation")

        except Exception as e:
            print(f"→ Key press demo issue: {type(e).__name__}: {e}")
            print(
                "  Hint: if navigation still occurs, the focused element may be in a submitting form."
            )
            print(
                "  Try manually focusing the textarea via browser dev tools to confirm."
            )

    def demo_mouse_press_release(self):
        """Shows press_mouse_on() + release_mouse_over() – manual drag simulation"""
        print("\n=== Demo: press_mouse_on() & release_mouse_over() ===\n")
        print("  (Simulates click-and-hold → drag → release)")
        try:
            source = helium.S("#drag1")  # adjust to real draggable id/class
            target = helium.S("#div1")  # drop zone

            if not (source.exists() and target.exists()):
                print("→ Drag source / target not found on this page")
                print("  (trytestingthis.netlify.app has a simple drag-drop area)")
                return

            print("→ Pressing mouse on source element")
            helium.press_mouse_on(source)
            time.sleep(1.2)

            print("→ Moving mouse (via JS fallback if needed)")
            # Helium does NOT have move_mouse_to() – we fake drag via JS if needed
            self.driver.execute_script(
                """
                arguments[0].dispatchEvent(new MouseEvent('mousemove', {
                    bubbles: true, cancelable: true, clientX: arguments[1], clientY: arguments[2]
                }));
            """,
                target.web_element,
                300,
                300,
            )  # rough approximation

            time.sleep(1)

            print("→ Releasing mouse over target")
            helium.release_mouse_over(target)
            time.sleep(1.8)

            print("→ Mouse press/release sequence finished")

            # Optional: alert / visual check
            if helium.Text("dropped").exists():
                print("→ Looks like drop succeeded!")

        except Exception as e:
            print(f"→ Mouse press/release demo failed: {e}")
            print("  Hint: this page may not have draggable elements → test elsewhere")

    def run_interactive_demos(self):
        """Run a sequence that demonstrates most actions"""
        print("Starting Helium actions demo sequence...\n")

        self.print_browser_state()

        self.demo_click()

        print("\nTyping credentials...")
        self.demo_login_form()  # ← more specific than just press ENTER

        self.print_browser_state()

        print("\nType first and last name...")
        self.demo_write()

        print("\nDropdown selection demo...")
        self.demo_select_dropdown()

        print("\nWaiting and existence checks...")
        self.demo_wait_until_and_exists()

        print("\nAdvanced selectors and relative locators...")
        self.demo_s_selector_and_relative()

        print("\nFinding multiple elements...")
        self.demo_find_all_elements()

        print("\nScrolling demo...")
        self.demo_scroll()

        print("\nFile upload on demo page...")
        self.demo_file_upload()

        print("\nCheckbox demo...")
        self.demo_checkbox()

        print("\nRadio button attempt (on main demo page – expect failures)...")
        self.demo_radio_button()

        print("\nLink element demo...")
        self.demo_link_element()

        print("\nImage element demo...")
        self.demo_image_element()

        print("\nDouble-click demo...")
        self.demo_double_click()

        print("\nDrag and drop demo...")
        self.demo_drag_and_drop()

        print("\nCoordinate click demo...")
        self.demo_point_click()

        print("\nHighlighting elements (visual debug)...")
        self.demo_highlight_element()

        print("\nAdvanced key presses (special keys / combos)...")
        self.demo_press_keys_advanced()

        print("\nFile attachment demo...")
        self.demo_attach_file()

        print("\nMouse press → hold → release demo...")
        self.demo_mouse_press_release()

    def run_read_demos(self):
        print("\nReading element values...")
        self.demo_read_values()

        print("\nReading links on page...")
        # Very fast even on Wikipedia
        self.demo_read_links(mode="smart", max_links=300, only_textual=True)

        # Original (slow but "nicest" visibility handling)
        # self.demo_read_links(mode="full")

        # Quick raw dump (includes icons, js links, etc.)
        self.demo_read_links(mode="fast")

        print("\nReading buttons on page...")
        self.demo_read_buttons()

        print("\nReading texts on page...")

        # Recommended for most real-world usage (fast + meaningful content)
        self.demo_read_texts(mode="smart", max_texts=500, min_length=5)

        # When you really need coordinates (debugging layout)
        # self.demo_read_texts(mode="full", max_texts=100)

        # Quick raw text dump
        self.demo_read_texts(mode="fast")

        print("\nReading list items on page...")

        # Most practical for large pages
        self.demo_read_list_items(mode="smart", max_items=400, min_length=5)

        # When debugging Helium behavior
        # self.demo_read_list_items(mode="full", max_items=100)

        # Quick raw check
        self.demo_read_list_items(mode="fast")

        print("\nTaking final screenshot...")
        self.demo_take_screenshot()
        self.demo_take_full_page_screenshot()
        self.demo_take_full_page_overflow_screenshot()

        print("\nDemo sequence finished.")


if __name__ == "__main__":
    base_output_dir = Path(__file__).parent / "generated" / Path(__file__).stem
    # Optional: print version for debugging
    # import helium
    # print("Helium version:", helium.__version__)

    # url = "https://trytestingthis.netlify.app"
    url = "https://en.wikipedia.org/wiki/Chicago"
    output_dir = base_output_dir / format_sub_source_dir(url)
    demo = DemoHeliumActions(
        url, headless=False, output_dir=output_dir
    )  # Change to True on server
    try:
        # demo.run_interactive_demos()
        demo.run_read_demos()
    finally:
        demo.close()
