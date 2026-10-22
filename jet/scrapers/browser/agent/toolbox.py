"""
Playwright Toolbox
Handles low-level browser interactions using Playwright API.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from jet.scrapers.browser.agent.utils import save_text
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
    ):
        self.headless = headless
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.auto_scroll = auto_scroll
        self.max_scrolls = max_scrolls
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.last_snapshot: Optional[str] = None
        self.last_url: Optional[str] = None  # Track last known URL for change detection
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
        # Initialize last_url
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
            # Reset last_url when creating new page
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

        # If we don't have a last_url recorded, consider it changed
        if self.last_url is None:
            return True

        # Compare URLs (normalize by removing trailing slashes for comparison)
        normalized_current = current_url.rstrip("/")
        normalized_last = self.last_url.rstrip("/")

        return normalized_current != normalized_last

    async def _update_last_url(self):
        """Update the last known URL."""
        self.last_url = await self._get_current_url()

    async def _scroll_to_bottom(self):
        """Internal helper to scroll to bottom to trigger lazy loading.

        Implements robust scrolling with:
        - Proper height detection using documentElement
        - Stabilization checks for lazy-loaded content
        - Configurable max attempts and delays
        - Bottom detection with reasonable threshold
        """
        if not self.auto_scroll or not self.page:
            return

        console.print(
            f"[dim]Auto-scrolling to bottom (max {self.max_scrolls} jumps)...[/dim]"
        )

        page = self.page

        # Get initial scroll height
        last_height = await page.evaluate("document.documentElement.scrollHeight")

        # Threshold for considering we're at the bottom (in pixels)
        BOTTOM_THRESHOLD = 100

        # Track consecutive scrolls without height change
        no_change_count = 0
        MAX_NO_CHANGE = 3

        for i in range(self.max_scrolls):
            # Scroll to bottom
            await page.evaluate(
                "window.scrollTo(0, document.documentElement.scrollHeight)"
            )

            # Wait for potential lazy-loaded content
            await page.wait_for_timeout(1000)

            # Check current metrics
            metrics = await page.evaluate("""
                () => {
                    return {
                        scrollHeight: document.documentElement.scrollHeight,
                        scrollY: window.scrollY,
                        innerHeight: window.innerHeight
                    };
                }
            """)

            new_height = metrics["scrollHeight"]
            view_bottom = metrics["scrollY"] + metrics["innerHeight"]

            # Check if we've reached the bottom
            reached_bottom = view_bottom >= (new_height - BOTTOM_THRESHOLD)

            # Check if height changed (indicating lazy-loaded content)
            height_changed = new_height != last_height

            console.print(
                f"[dim]Scroll {i + 1}/{self.max_scrolls}: "
                f"height={new_height}px, "
                f"viewBottom={view_bottom}px, "
                f"bottom={reached_bottom}, "
                f"changed={height_changed}[/dim]"
            )

            # If height hasn't changed, increment counter
            if not height_changed:
                no_change_count += 1
            else:
                no_change_count = 0

            # Stop conditions:
            # 1. Reached bottom AND height is stable
            # 2. Height hasn't changed for MAX_NO_CHANGE consecutive scrolls
            if reached_bottom and not height_changed:
                console.print("[dim]Reached bottom with stable content[/dim]")
                break
            elif no_change_count >= MAX_NO_CHANGE:
                console.print(
                    f"[yellow]Height stable for {no_change_count} scrolls, stopping[/yellow]"
                )
                break

            # Update last height for next iteration
            last_height = new_height
        else:
            console.print(
                f"[yellow]Reached max scrolls ({self.max_scrolls}) without stabilization[/yellow]"
            )

    async def _maybe_scroll_if_url_changed(self):
        """Check if URL has changed and scroll if needed.

        This should be called after any action that might cause navigation.
        """
        if await self._has_url_changed():
            console.print("[dim]URL changed detected, triggering scroll[/dim]")
            await self._scroll_to_bottom()
            await self._update_last_url()

    async def browser_navigate(self, url: str):
        page = await self._ensure_page()
        console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
        await page.goto(url)
        self.last_snapshot = None
        # Update URL tracking
        await self._update_last_url()
        # Always scroll after explicit navigation
        await self._scroll_to_bottom()

    async def browser_set_content(self, html: str):
        page = await self._ensure_page()
        console.print("[bold blue]Setting page content from HTML string...[/bold blue]")
        await page.set_content(html)
        self.last_snapshot = None
        # Update URL tracking (content change may reset URL)
        await self._update_last_url()
        # Always scroll after setting content
        await self._scroll_to_bottom()

    async def browser_navigate_back(self):
        page = await self._ensure_page()
        await page.go_back()
        self.last_snapshot = None
        # Update URL tracking
        await self._update_last_url()
        # Always scroll after navigation
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
        save_text(save_path, snapshot)
        console.print(f"[green]Snapshot saved:[/green] {save_path.name}")
        try:
            html_content = await page.content()
            html_path = (step_dir or self.output_dir) / "snapshot.html"
            save_text(html_path, html_content)
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
        await locator.wait_for(state="visible", timeout=5000)
        if doubleClick:
            await locator.dblclick(button=button)
        else:
            await locator.click(button=button)
        console.print(f"[green]Clicked {target}[/green]")
        # Check if click caused navigation and scroll if needed
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
        # Check if typing/submit caused navigation and scroll if needed
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
        # Form filling typically doesn't navigate, but check anyway
        await self._maybe_scroll_if_url_changed()

    async def browser_wait_for(
        self, time: Optional[float] = None, text: Optional[str] = None
    ):
        page = await self._ensure_page()
        if time:
            await page.wait_for_timeout(time * 1000)
        elif text:
            await page.get_by_text(text).first.wait_for(state="visible")
        # Waiting shouldn't cause navigation, but check just in case
        await self._maybe_scroll_if_url_changed()

    async def browser_press_key(self, key: str):
        page = await self._ensure_page()
        await page.keyboard.press(key)
        console.print(f"[green]Pressed key: {key}[/green]")
        # Key press might cause navigation (e.g., Enter on a link)
        await self._maybe_scroll_if_url_changed()

    async def browser_hover(self, target: str):
        page = await self._ensure_page()
        resolved = self._resolve_target(target)
        await page.locator(resolved).first.hover()
        console.print(f"[green]Hovered {target}[/green]")
        # Hover shouldn't cause navigation

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
        # Select option typically doesn't navigate

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
        if target:
            resolved = self._resolve_target(target)
            result = await page.locator(resolved).first.evaluate(function)
        else:
            result = await page.evaluate(function)
        console.print(f"[green]Evaluated JS. Result: {result}[/green]")
        # JS evaluation might cause navigation
        await self._maybe_scroll_if_url_changed()
        return result

    async def browser_run_code_unsafe(self, code: str):
        page = await self._ensure_page()
        script = f"(async () => {{ {code} }})()"
        result = await page.evaluate(script)
        console.print(f"[green]Executed custom code. Result: {result}[/green]")
        # Custom code might cause navigation
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
        console.print(f"[green]Screenshot saved:[/green] {output_path.name}")
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
        save_text(answer_path, save_content)
        console.print(f"[green]Answer saved:[/green] {answer_path.name}")
        return result
