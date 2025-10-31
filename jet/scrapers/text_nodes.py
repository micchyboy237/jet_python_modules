import os
import re
import uuid
import base64

from typing import Optional, List
from pyquery import PyQuery as pq
from lxml.etree import Comment
from fake_useragent import UserAgent
from playwright.sync_api import sync_playwright

from jet.scrapers.browser.config import PLAYWRIGHT_CHROMIUM_EXECUTABLE
from jet.scrapers.utils import BaseNode, exclude_elements, get_xpath
from jet.utils.inspect_utils import get_entry_file_path
from jet.transformers.formatters import format_html
from jet.scrapers.config import TEXT_ELEMENTS
from jet.search.formatters import decode_text_with_unidecode
from jet.logger import logger


def extract_nodes(element, depth: int = 0, parent_id: Optional[str] = None) -> List[BaseNode]:
    nodes = []
    element_pq = pq(element)
    text = element_pq.text().strip() if element_pq.text() else ""
    tag = element_pq[0].tag if element_pq else "unknown"
    # Skip comment nodes
    if tag is Comment or str(tag).startswith('<cyfunction Comment'):
        return nodes
    tag = tag.lower() if isinstance(tag, str) else "unknown"
    if tag == "html":
        for child in element_pq.children():
            nodes.extend(extract_nodes(child, depth + 1, parent_id))
        return nodes
    valid_id_pattern = r'^[a-zA-Z0-9_-]+$'
    element_id = element_pq.attr('id')
    element_class = element_pq.attr('class')
    adjusted_depth = max(1, depth - 2) if depth > 2 else 1
    id = element_id if element_id and re.match(
        valid_id_pattern, element_id) else f"node_{adjusted_depth}_{len(nodes)}"
    class_names = [name for name in (element_class.split() if element_class else [])
                   if re.match(valid_id_pattern, name)]
    if text and len(element_pq.children()) == 0:
        nodes.append(BaseNode(
            tag=tag,
            text=text,
            depth=adjusted_depth,
            raw_depth=depth,
            id=id,
            class_names=class_names,
            line=element_pq[0].sourceline if element_pq else 0,
            html=element_pq.outer_html()
        ))
    for child in element_pq.children():
        nodes.extend(extract_nodes(child, depth + 1, id))
    return nodes


def extract_text_nodes(
    source: str,
    excludes: List[str] = ["nav", "footer", "script", "style"],
    timeout_ms: int = 10000,
    with_screenshot: bool = True,
    wait_for_js: bool = True,
    headless: bool = True,
) -> List[BaseNode]:
    """
    Extracts text nodes from an HTML source using Playwright, excluding specified tags.
    Args:
        source: The HTML string, file path, or URL to parse.
        excludes: A list of tag names to exclude (e.g., ["nav", "footer", "script", "style"]).
        timeout_ms: Timeout for rendering the page (in ms) for dynamic content.
        with_screenshot: Whether to capture screenshots during rendering.
        wait_for_js: Whether to wait for JavaScript execution to complete.
        headless: Whether to run browser in headless mode.
    Returns:
        A list of BaseNode objects containing text, tag, depth, id, class_names, line number, and HTML.
    """
    if os.path.exists(source) and not source.startswith("file://"):
        source = f"file://{source}"
    
    if re.match(r'^https?://', source) or re.match(r'^file://', source):
        url = source
        html = None
    else:
        url = None
        html = source
    
    with sync_playwright() as p:
        traces_dir = f"{get_entry_file_path(remove_extension=True)}/generated/playwright/traces"
        os.makedirs(traces_dir, exist_ok=True)
        
        browser = p.chromium.launch(
            headless=headless,
            executable_path=PLAYWRIGHT_CHROMIUM_EXECUTABLE,
            traces_dir=traces_dir,
        )
        
        ua = UserAgent()
        page = browser.new_page(user_agent=ua.random)
        
        if url:
            page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
        else:
            page.set_content(html, timeout=timeout_ms, wait_until="domcontentloaded")
        
        if wait_for_js:
            js_timeout = 5000
            logger.debug(f"Waiting JS content for {js_timeout // 1000}s")
            page.wait_for_timeout(js_timeout)
        
        if with_screenshot:
            # Generate a random screenshot name using uuid
            screenshot_name = f"screenshot_{uuid.uuid4().hex}.png"
            screenshot_path = f"{get_entry_file_path(remove_extension=True)}/generated/playwright/screenshots/{screenshot_name}"
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            screenshot = page.screenshot(full_page=True, path=screenshot_path)
            if screenshot:
                decoded_screenshot = base64.b64encode(screenshot).decode('utf-8')
                logger.debug(f"Decoded screenshot, length: {len(decoded_screenshot)}")
                logger.success(f"Screenshot saved at: {screenshot_path}")
        
        page_content = page.content()
        browser.close()
    
    # Format the HTML content using format_html
    formatted_html = format_html(page_content)
    doc = pq(formatted_html)
    exclude_elements(doc, excludes)
    
    # Split formatted HTML into lines for line number tracking
    html_lines = formatted_html.splitlines()
    
    text_nodes = []
    stack = [(doc[0], 0)]  # (element, depth)
    
    while stack:
        el, depth = stack.pop()
        el_pq = pq(el)
        
        for child in el_pq.children():
            child_pq = pq(child)
            
            # Skip comment nodes
            if child.tag is Comment or str(child.tag).startswith('<cyfunction Comment'):
                continue
            
            tag = child.tag if isinstance(child.tag, str) else str(child.tag)
            class_names = [cls for cls in (child_pq.attr("class") or "").split() 
                          if not cls.startswith("css-")]
            element_id = child_pq.attr("id")
            
            if not element_id or not re.match(r'^[a-zA-Z_-]+$', element_id):
                element_id = f"auto_{uuid.uuid4().hex[:8]}"
            
            # Handle text extraction based on tag type
            if tag.lower() == "meta":
                text = child_pq.attr("content") or ""
                text = decode_text_with_unidecode(text.strip())
            elif tag.lower() in TEXT_ELEMENTS:
                text_parts = []
                for item in child_pq.contents():
                    if isinstance(item, str):
                        text_parts.append(item.strip())
                    else:
                        text_parts.append(decode_text_with_unidecode(
                            pq(item).text().strip()))
                text = " ".join(part for part in text_parts if part)
            else:
                text = decode_text_with_unidecode(child_pq.text().strip())
            
            # Get the actual line number from lxml's sourceline if available
            line_number = getattr(child, 'sourceline', None)
            if line_number is None:
                # Fallback: Approximate line number by searching for the element's tag
                element_str = f"<{tag}"
                for i, line in enumerate(html_lines, 1):
                    if element_str in line:
                        line_number = i
                        break
                else:
                    line_number = 1  # Default fallback
            
            # Create BaseNode (assuming BaseNode has similar fields to TreeNode)
            xpath = get_xpath(child)
            node = BaseNode(
                tag=tag,
                text=text,
                depth=depth + 1,
                id=element_id,
                class_names=class_names,
                line=line_number,
                xpath=xpath,
                html=child_pq.outer_html()
            )
            
            # Only add nodes with meaningful text content
            if text.strip():
                text_nodes.append(node)
            
            # Traverse deeper if the tag is not a text element and has children
            if tag.lower() not in TEXT_ELEMENTS and child_pq.children():
                stack.append((child, depth + 1))
    
    return text_nodes