"""
Playwright Toolbox
Handles low-level browser interactions using Playwright API.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jet.scrapers.browser.agent.utils import save_json, save_text
from jet.scrapers.html_rag_extractor import extract_content
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from rich.console import Console

console = Console()

ACTION_TIMEOUT = 10000
NAVIGATION_TIMEOUT = 15000


class PlaywrightToolbox:
    def __init__(
        self,
        headless: bool = False,
        output_dir: str | Path = None,
        auto_scroll: bool = True,
        max_scrolls: int = 5,
        scroll_delay_ms: int = 1400,
        scroll_timeout_ms: int = 45000,
        scroll_mode: Literal["jump", "increment"] = "increment",
    ):
        self.headless = headless
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.auto_scroll = auto_scroll
        self.max_scrolls = max_scrolls
        self.scroll_delay_ms = scroll_delay_ms
        self.scroll_timeout_ms = scroll_timeout_ms
        self.scroll_mode = scroll_mode
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.last_snapshot: Optional[str] = None
        self.last_url: Optional[str] = None
        self.is_closed = False

    async def start(self):
        console.print("[bold blue]Initializing Playwright...[/bold blue]")
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1280, "height": 720}
        )
        self.page = await self.context.new_page()
        self.page.set_default_timeout(ACTION_TIMEOUT)
        self.page.set_default_navigation_timeout(NAVIGATION_TIMEOUT)
        self.last_url = self.page.url if self.page else None
        console.print("[green]Browser ready[/green]")

    async def close(self):
        if self.is_closed:
            return
        self.is_closed = True
        for obj in [self.page, self.context, self.browser, self.playwright]:
            if obj:
                try:
                    await obj.close() if hasattr(obj, "close") else await obj.stop()
                except:
                    pass
        console.print("[green]Browser closed[/green]")

    async def _ensure_page(self) -> Page:
        if self.is_closed:
            raise Exception("Browser is closed")
        if not self.page or self.page.is_closed():
            self.page = await self.context.new_page()
            self.page.set_default_timeout(ACTION_TIMEOUT)
            self.last_url = self.page.url if self.page else None
        return self.page

    def _resolve_target(self, target: str) -> str:
        if self.last_snapshot and re.match(r"^[a-zA-Z0-9_]+$", target):
            pattern = rf"- (\w+) \"([^\"]+)\" \[ref={re.escape(target)}\]"
            match = re.search(pattern, self.last_snapshot)
            if match:
                role, name = match.groups()
                return f"internal:role={role}[name='{name}'i]"
        if re.match(r"^[a-zA-Z0-9_]+$", target):
            return f"aria-ref={target}"
        return target

    async def _get_current_url(self) -> str:
        """Safely get current page URL."""
        try:
            if self.page and not self.page.is_closed():
                return self.page.url
        except:
            pass
        return "Unknown"

    async def _has_url_changed(self) -> bool:
        """Check if the current URL has changed from the last known URL."""
        current_url = await self._get_current_url()
        if current_url == "Unknown":
            return False
        if self.last_url is None:
            return True
        normalized_current = current_url.rstrip("/")
        normalized_last = self.last_url.rstrip("/")
        return normalized_current != normalized_last

    async def _update_last_url(self):
        """Update the last known URL."""
        self.last_url = await self._get_current_url()

    async def _scroll_to_bottom(
        self,
        strategy: Literal["none", "once", "until_stable"] = "until_stable",
        mode: Optional[Literal["jump", "increment"]] = None,
        max_attempts: Optional[int] = None,
        delay_ms: Optional[int] = None,
        overall_timeout_ms: Optional[int] = None,
    ):
        """Scroll to bottom of page with various strategies."""
        if not self.auto_scroll or not self.page or strategy == "none":
            return

        # Use instance defaults if not overridden
        mode = mode or self.scroll_mode
        max_attempts = max_attempts or self.max_scrolls
        delay_ms = delay_ms or self.scroll_delay_ms
        overall_timeout_ms = overall_timeout_ms or self.scroll_timeout_ms

        console.print(
            f"[dim]Auto-scrolling to bottom (strategy={strategy}, mode={mode}, max={max_attempts})...[/dim]"
        )

        page = self.page
        start_time = asyncio.get_event_loop().time()

        # Initial check for document readiness
        try:
            last_height = await page.evaluate("document.documentElement.scrollHeight")
        except Exception as e:
            console.print(f"[yellow]Scroll init failed: {e}[/yellow]")
            return

        STABILIZE_THRESHOLD_PX = 150
        no_change_count = 0
        MAX_NO_CHANGE = 4

        for attempt in range(1, max_attempts + 1):
            # Check overall timeout
            if (
                asyncio.get_event_loop().time() - start_time
            ) * 1000 > overall_timeout_ms:
                console.print("[yellow]Scroll timeout reached[/yellow]")
                break

            try:
                if mode == "jump":
                    await page.evaluate(
                        "window.scrollTo(0, document.documentElement.scrollHeight)"
                    )
                else:
                    await page.evaluate("window.scrollBy(0, window.innerHeight * 0.9)")

                await page.wait_for_timeout(delay_ms)

                if strategy == "once":
                    break

                # Safe evaluation with null checks
                metrics = await page.evaluate("""() => {
                    if (!document.documentElement) return { scrollHeight: 0, scrollY: 0, innerHeight: 0 };
                    return {
                        scrollHeight: document.documentElement.scrollHeight,
                        scrollY: window.scrollY,
                        innerHeight: window.innerHeight
                    };
                }""")

                new_height = metrics["scrollHeight"]
                view_bottom = metrics["scrollY"] + metrics["innerHeight"]
                bottom_reached = view_bottom >= (new_height - STABILIZE_THRESHOLD_PX)
                height_changed = new_height != last_height

                console.print(
                    f"[dim]Scroll {attempt}/{max_attempts}: "
                    f"height={new_height}px, "
                    f"viewBottom={view_bottom}px, "
                    f"bottom={bottom_reached}, "
                    f"changed={height_changed}[/dim]"
                )

                if not height_changed:
                    no_change_count += 1
                else:
                    no_change_count = 0

                if mode == "increment":
                    if bottom_reached:
                        if height_changed:
                            console.print(
                                "[dim]Near bottom but height still changing → continue[/dim]"
                            )
                        else:
                            console.print(
                                f"[dim]Reached near bottom and height stable after {attempt} attempts[/dim]"
                            )
                            break
                    elif no_change_count >= MAX_NO_CHANGE:
                        console.print(
                            f"[yellow]Height not changing for {no_change_count} attempts, but not at bottom yet → stopping[/yellow]"
                        )
                        break
                else:  # jump mode
                    if not height_changed:
                        console.print(
                            f"[dim]Scroll stabilized (height only) after {attempt} attempts[/dim]"
                        )
                        break

                last_height = new_height

            except Exception as e:
                console.print(f"[yellow]Scroll step {attempt} failed: {e}[/yellow]")
                break
        else:
            console.print(
                f"[yellow]Reached max scrolls ({max_attempts}) without stabilization[/yellow]"
            )

    async def _maybe_scroll_if_url_changed(self):
        """Check if URL has changed and scroll if needed."""
        if await self._has_url_changed():
            console.print("[dim]URL changed detected, triggering scroll[/dim]")
            await self._scroll_to_bottom()
            await self._update_last_url()

    async def browser_navigate(self, url: str):
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        self.last_snapshot = None
        await self._update_last_url()
        await self._scroll_to_bottom()

    async def browser_set_content(self, html: str):
        page = await self._ensure_page()
        console.print("[bold blue]Setting page content from HTML string...[/bold blue]")
        await page.set_content(html)
        self.last_snapshot = None
        await self._update_last_url()
        await self._scroll_to_bottom()

    async def browser_navigate_back(self):
        page = await self._ensure_page()
        await page.go_back()
        self.last_snapshot = None
        await self._update_last_url()
        await self._scroll_to_bottom()

    async def browser_snapshot(
        self,
        filename: Optional[str] = None,
        depth: Optional[int] = None,
        boxes: Optional[bool] = False,
        step_dir: Optional[Path] = None,
    ):
        page = await self._ensure_page()
        snapshot = await page.aria_snapshot(mode="ai", depth=depth, boxes=boxes)
        self.last_snapshot = snapshot
        save_path = (step_dir or self.output_dir) / (filename or "snapshot.md")
        link = save_text(save_path, snapshot)
        console.print(f"[green]Snapshot saved:[/green] {link}")
        try:
            html_content = await page.content()
            html_path = (step_dir or self.output_dir) / "snapshot.html"
            html_link = save_text(html_path, html_content)
            console.print(f"[dim]HTML saved:[/dim] {html_link}")
        except Exception as e:
            console.print(f"[yellow]Failed to save HTML snapshot: {e}[/yellow]")
        return snapshot

    async def browser_click(
        self,
        target: str,
        doubleClick: bool = False,
        button: Literal["left", "right", "middle"] = "left",
    ):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first

        # Wait for visibility
        await locator.wait_for(state="visible", timeout=5000)

        # Force scroll into view to prevent "outside viewport" errors
        try:
            await locator.scroll_into_view_if_needed(timeout=2000)
        except:
            pass  # Ignore if it fails, Playwright click usually handles it

        if doubleClick:
            await locator.dblclick(button=button)
        else:
            await locator.click(button=button)

        console.print(f"[green]Clicked {target}[/green]")
        await self._maybe_scroll_if_url_changed()

    async def browser_type(self, target: str, text: str, submit: bool = False):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        locator = page.locator(resolved).first
        await locator.wait_for(state="visible", timeout=5000)
        await locator.fill(text)
        if submit:
            await locator.press("Enter")
        console.print(f"[green]Typed '{text}' into {target}[/green]")
        await self._maybe_scroll_if_url_changed()

    async def browser_fill_form(self, fields: List[Dict[str, str]]):
        page = await self._ensure_page()
        console.print(
            f"[bold blue]Filling form with {len(fields)} fields...[/bold blue]"
        )
        for field_data in fields:
            target, value = field_data.get("target"), field_data.get("value", "")
            if not target:
                continue
            resolved = self._resolve_target(target)
            locator = page.locator(resolved).first
            await locator.wait_for(state="visible", timeout=5000)
            await locator.fill(value)
            console.print(f"[green]Filled {target} with '{value}'[/green]")
        await self._maybe_scroll_if_url_changed()

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None
    ):
        page = await self._ensure_page()
        if time:
            await page.wait_for_timeout(time * 1000)
        elif text:
            await page.get_by_text(text).first.wait_for(state="visible")
        await self._maybe_scroll_if_url_changed()

    async def browser_press_key(self, key: str):
        page = await self._ensure_page()
        await page.keyboard.press(key)
        console.print(f"[green]Pressed key: {key}[/green]")
        await self._maybe_scroll_if_url_changed()

    async def browser_hover(self, target: str):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        await page.locator(resolved).first.hover()
        console.print(f"[green]Hovered {target}[/green]")

    async def browser_extract_content(self, step_dir: Optional[Path] = None):
        """
        Extracts the main readable content from the current page HTML using Trafilatura.
        Returns the content as Markdown with shortened URLs.
        """
        page = await self._ensure_page()
        try:
            html_content = await page.content()
            current_url = await self._get_current_url()
            markdown_content, url_map = extract_content(html_content, url=current_url)
            if step_dir:
                md_link = save_text(step_dir / "markdown_content.md", markdown_content)
                map_link = save_json(step_dir / "url_map.json", url_map)

                console.print(
                    f"[green]Markdown saved ({len(markdown_content)} chars):[/green] {md_link}"
                )
                console.print(
                    f"[green]URL Map saved ({len(url_map)} links shortened):[/green] {map_link}"
                )
            else:
                console.print(
                    f"[green]Extracted content ({len(markdown_content)} chars, {len(url_map)} links shortened)[/green]"
                )
            return markdown_content
        except Exception as e:
            console.print(f"[bold red]Error extracting content:[/bold red] {e}")
            raise e

    async def browser_find(
        self,
        text: Optional[str] = None,
        regex: Optional[str] = None,
        maxResults: Optional[int] = None,
    ):
        if self.last_snapshot:
            console.print(f"[dim]Searching snapshot for: {text or regex}[/dim]")
            return "Search completed in snapshot."
        return "No snapshot available."

    async def browser_select_option(self, target: str, values: List[str]):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        await page.locator(resolved).first.select_option(values)
        console.print(f"[green]Selected options {values} in {target}[/green]")

    async def browser_handle_dialog(
        self, accept: bool, promptText: Optional[str] = None
    ):
        page = await self._ensure_page()

        async def handle_dialog(dialog):
            await dialog.accept(promptText) if accept else await dialog.dismiss()

        page.once("dialog", handle_dialog)
        console.print(f"[green]Set dialog handler: accept={accept}[/green]")

    async def browser_evaluate(self, function: str, target: Optional[str] = None):
        page = await self._ensure_page()

        # Sanitize and wrap the function string if necessary
        # Playwright's evaluate expects a function or an expression.
        # If the LLM returns "return document.title", it fails.
        # We wrap it in an arrow function if it starts with 'return' or contains top-level returns.
        js_code = function.strip()
        if js_code.startswith("return ") or (
            js_code.startswith("{") and "return " in js_code
        ):
            # If it looks like a function body, wrap it
            js_code = f"() => {{ {js_code} }}"
        elif not js_code.startswith("(") and not js_code.startswith("function"):
            # If it's just an expression like "document.title", wrap it
            js_code = f"() => {{ return {js_code}; }}"

        try:
            if target:
                resolved = self._resolve_target(target)
                locator = page.locator(resolved).first
                await locator.wait_for(state="visible", timeout=5000)
                result = await locator.evaluate(js_code)
            else:
                result = await page.evaluate(js_code)

            console.print(f"[green]Evaluated JS. Result: {result}[/green]")
            await self._maybe_scroll_if_url_changed()
            return result
        except Exception as e:
            console.print(f"[bold red]JS Evaluation Error:[/bold red] {e}")
            raise

    async def browser_run_code_unsafe(self, code: str):
        page = await self._ensure_page()
        script = f"(async () => {{ {code} }})()"
        result = await page.evaluate(script)
        console.print(f"[green]Executed custom code. Result: {result}[/green]")
        await self._maybe_scroll_if_url_changed()
        return result

    async def browser_take_screenshot(
        self,
        filename: Optional[str] = None,
        fullPage: bool = True,
        type: str = "png",
        step_dir: Optional[Path] = None,
    ):
        page = await self._ensure_page()
        base_name = filename or "screenshot"
        if not base_name.endswith(f".{type}"):
            base_name = f"{Path(base_name).stem}.{type}"
        output_path = (step_dir or self.output_dir) / base_name
        await page.screenshot(path=str(output_path), full_page=fullPage, type=type)

        # Use the standard Rich link format
        abs_path = output_path.resolve()
        link = f"[link=file://{abs_path}]{output_path.name}[/link]"
        console.print(f"[green]Screenshot saved:[/green] {link}")
        return str(output_path)

    async def browser_close(self):
        await self.close()

    async def final_answer(self, result: Any):
        console.print(f"[bold green]Final Answer Received[/bold green]")
        file_ext = "json" if not isinstance(result, str) else "txt"
        save_content = (
            json.dumps(result, indent=2) if not isinstance(result, str) else result
        )
        console.print(save_content)
        answer_path = self.output_dir / f"final_answer.{file_ext}"
        link = save_text(answer_path, save_content)
        console.print(f"[green]Answer saved:[/green] {link}")
        return result
