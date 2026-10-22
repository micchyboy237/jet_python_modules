"""
Playwright ARIA Snapshot Generator
Captures ARIA snapshots from live URLs or local HTML files using Playwright.
"""

import re
from pathlib import Path
from typing import Literal, Set, Union

from rich.console import Console

from playwright.async_api import async_playwright

console = Console()

# Global Literal Variable for Filter Types
FilterType = Literal[
    "all", "interactives", "non-interactives", "links", "rag", "fields"
]

# Comprehensive ARIA Roles based on W3C Spec
INTERACTIVE_ROLES: Set[str] = {
    "button",
    "checkbox",
    "combobox",
    "gridcell",
    "link",
    "listbox",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "radio",
    "searchbox",
    "slider",
    "spinbutton",
    "switch",
    "tab",
    "textbox",
    "treeitem",
    "menubutton",
    "toolbarbutton",
}

NON_INTERACTIVE_ROLES: Set[str] = {
    "article",
    "banner",
    "blockquote",
    "caption",
    "cell",
    "code",
    "columnheader",
    "complementary",
    "contentinfo",
    "definition",
    "deletion",
    "directory",
    "document",
    "emphasis",
    "feed",
    "figure",
    "form",
    "generic",
    "grid",
    "group",
    "heading",
    "img",
    "insertion",
    "list",
    "listitem",
    "log",
    "main",
    "marquee",
    "math",
    "menu",
    "menubar",
    "meter",
    "navigation",
    "none",
    "note",
    "paragraph",
    "presentation",
    "progressbar",
    "radiogroup",
    "region",
    "row",
    "rowgroup",
    "rowheader",
    "scrollbar",
    "search",
    "separator",
    "status",
    "strong",
    "subscript",
    "superscript",
    "table",
    "tablist",
    "tabpanel",
    "term",
    "time",
    "timer",
    "tooltip",
    "tree",
    "treegrid",
    "alert",
    "alertdialog",
    "application",
    "dialog",
}

# Roles optimized for LLM RAG (Content-heavy roles)
RAG_ROLES: Set[str] = {
    "article",
    "heading",
    "paragraph",
    "list",
    "listitem",
    "blockquote",
    "code",
    "definition",
    "term",
    "figure",
    "caption",
    "main",
    "region",
    "section",
    "generic",
    "text",
    "strong",
    "emphasis",
    "note",
    "doc-abstract",
    "doc-bibliography",
    "doc-chapter",
    "doc-conclusion",
    "doc-introduction",
}

# Roles optimized for Form Filling (Input-heavy roles)
FIELD_ROLES: Set[str] = {
    "textbox",
    "checkbox",
    "radio",
    "combobox",
    "listbox",
    "slider",
    "spinbutton",
    "searchbox",
    "switch",
    "button",
    "form",
    "link",
}


def _get_indent_level(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _get_role(line: str) -> str:
    match = re.search(r"^\s*-\s+(\w+)", line)
    return match.group(1).lower() if match else ""


def _is_relevant(line: str, filter_type: FilterType) -> bool:
    """Check if a line is relevant based on the filter type."""
    if not line.strip():
        return False
    role = _get_role(line)
    if not role:
        # Lines without roles are usually attributes (like /url) or text content
        # We generally want to keep these if their parent is kept.
        return True
    if filter_type == "interactives":
        return role in INTERACTIVE_ROLES
    elif filter_type == "non-interactives":
        return role in NON_INTERACTIVE_ROLES
    elif filter_type == "links":
        return role == "link"
    elif filter_type == "rag":
        return role in RAG_ROLES
    elif filter_type == "fields":
        return role in FIELD_ROLES
    return False


def filter_aria_snapshot(snapshot: str, filter_type: FilterType) -> str:
    if filter_type == "all":
        return snapshot

    lines = snapshot.split("\n")
    if not lines:
        return snapshot

    # 1. Identify matched indices (lines that explicitly match the role criteria)
    matched_indices: Set[int] = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        role = _get_role(line)
        if not role:
            continue

        if _is_relevant(line, filter_type):
            matched_indices.add(i)

    if not matched_indices:
        return ""

    # 2. Expand selection to include Context (Parents and Relevant Children)
    final_indices: Set[int] = set()

    for idx in matched_indices:
        # A. Add the matched line itself
        final_indices.add(idx)

        # B. Add Relevant Children
        # We only add children if they are also relevant OR if they are attribute lines (no role)
        current_indent = _get_indent_level(lines[idx])
        j = idx + 1
        while j < len(lines):
            if not lines[j].strip():
                j += 1
                continue
            next_indent = _get_indent_level(lines[j])

            if next_indent > current_indent:
                # It's a child. Is it relevant?
                if _is_relevant(lines[j], filter_type):
                    final_indices.add(j)
                elif not _get_role(lines[j]):
                    # It's an attribute line (e.g., /url), keep it
                    final_indices.add(j)
                j += 1
            else:
                break

        # C. Add Parents (walk up the tree)
        parent_indent = current_indent - 2
        while parent_indent >= 0:
            parent_idx = None
            for k in range(idx - 1, -1, -1):
                if not lines[k].strip():
                    continue
                k_indent = _get_indent_level(lines[k])
                if k_indent == parent_indent:
                    parent_idx = k
                    break
                elif k_indent < parent_indent:
                    break

            if parent_idx is not None:
                final_indices.add(parent_idx)
                # Move up to the next parent level
                idx = parent_idx
                current_indent = parent_indent
                parent_indent -= 2
            else:
                break

    # 3. Construct the filtered snapshot
    filtered_lines = []
    for i, line in enumerate(lines):
        if i in final_indices:
            filtered_lines.append(line)

    return "\n".join(filtered_lines)


async def _capture_snapshot_from_url(
    url: str,
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
    filter_type: FilterType = "all",
) -> str:
    console.print(f"[bold blue]Starting snapshot capture for URL:[/bold blue] {url}")
    console.print(
        f"[dim]Parameters: timeout={timeout}ms, depth={depth}, mode={mode}, boxes={boxes}, headless={headless}, filter={filter_type}[/dim]"
    )
    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            console.print(f"[bold blue]Navigating to:[/bold blue] {url}")
            await page.goto(url, timeout=timeout)
            console.print("[green]Page loaded successfully[/green]")
            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")
            snapshot = await _take_snapshot(page, timeout, depth, mode, boxes)
            return filter_aria_snapshot(snapshot, filter_type)
        except Exception as e:
            console.print(f"[bold red]Error during snapshot capture:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


async def _capture_snapshot_from_html(
    html_string: str,
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
    filter_type: FilterType = "all",
) -> str:
    console.print("[bold blue]Starting snapshot capture from HTML string[/bold blue]")
    console.print(
        f"[dim]Parameters: timeout={timeout}ms, depth={depth}, mode={mode}, boxes={boxes}, headless={headless}, filter={filter_type}[/dim]"
    )
    async with async_playwright() as p:
        console.print("[bold blue]Launching browser...[/bold blue]")
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page()
        try:
            console.print("[bold blue]Loading HTML content...[/bold blue]")
            await page.set_content(html_string)
            console.print("[green]HTML content loaded successfully[/green]")
            await page.wait_for_load_state("networkidle", timeout=timeout)
            console.print("[dim]Page network idle state reached[/dim]")
            snapshot = await _take_snapshot(page, timeout, depth, mode, boxes)
            return filter_aria_snapshot(snapshot, filter_type)
        except Exception as e:
            console.print(f"[bold red]Error during snapshot capture:[/bold red] {e}")
            raise
        finally:
            console.print("[bold blue]Closing browser...[/bold blue]")
            await browser.close()
            console.print("[green]Browser closed[/green]")


async def _take_snapshot(
    page,
    timeout: float,
    depth: int | None,
    mode: Literal["ai", "default"],
    boxes: bool,
) -> str:
    console.print("[bold blue]Capturing ARIA snapshot...[/bold blue]")
    snapshot = await page.aria_snapshot(
        timeout=timeout,
        depth=depth,
        mode=mode,
        boxes=boxes,
    )
    console.print("[green]ARIA snapshot captured successfully[/green]")
    console.print(f"[dim]Snapshot size: {len(snapshot)} characters[/dim]")
    return snapshot


async def capture_snapshot(
    source: Union[str, Path],
    timeout: float = 30000,
    depth: int | None = None,
    mode: Literal["ai", "default"] = "ai",
    boxes: bool = False,
    headless: bool = False,
    filter_type: FilterType = "all",
) -> str:
    source_path = Path(source)
    if source_path.is_file():
        console.print(f"[dim]Detected local file: {source_path}[/dim]")
        with open(source_path, "r", encoding="utf-8") as f:
            html = f.read()
        return await _capture_snapshot_from_html(
            html_string=html,
            timeout=timeout,
            depth=depth,
            mode=mode,
            boxes=boxes,
            headless=headless,
            filter_type=filter_type,
        )
    else:
        console.print(f"[dim]Detected URL: {source}[/dim]")
        return await _capture_snapshot_from_url(
            url=str(source),
            timeout=timeout,
            depth=depth,
            mode=mode,
            boxes=boxes,
            headless=headless,
            filter_type=filter_type,
        )
