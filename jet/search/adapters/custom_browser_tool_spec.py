from typing import Optional, List
from playwright.async_api import Browser as AsyncBrowser, Page
from llama_index.core.tools.tool_spec.base import BaseToolSpec

# Default configuration values
DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS = 900
DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS = 300
DEFAULT_WAIT_FOR_NETWORK_IDLE = True
DEFAULT_INCLUDE_HIDDEN_DATA = True
DEFAULT_INCLUDE_HIDDEN_ELEMENTS = False
DEFAULT_RESPONSE_MODE = "fast"
DEFAULT_CONTENT_SELECTORS = ["body", "main", "article", "[role='main']"]
DEFAULT_EXCLUDE_SELECTORS = [
    "script", "style", "nav", "footer", "[class*='ad']"]

# Error messages
QUERY_PROMPT_REQUIRED_ERROR_MESSAGE = "Either query or prompt must be provided."
QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE = "Cannot provide both query and prompt."


class CustomBrowserToolSpec(BaseToolSpec):
    """
    Custom Browser Tool Specification for extracting data and elements from web pages.
    Replaces AgentQLBrowserToolSpec with a reusable, dependency-free implementation.
    """
    spec_functions = [
        "extract_web_data_from_browser",
        "get_web_element_from_browser",
    ]

    def __init__(
        self,
        async_browser: AsyncBrowser,
        timeout_for_data: int = DEFAULT_EXTRACT_DATA_TIMEOUT_SECONDS,
        timeout_for_element: int = DEFAULT_EXTRACT_ELEMENTS_TIMEOUT_SECONDS,
        wait_for_network_idle: bool = DEFAULT_WAIT_FOR_NETWORK_IDLE,
        include_hidden_for_data: bool = DEFAULT_INCLUDE_HIDDEN_DATA,
        include_hidden_for_element: bool = DEFAULT_INCLUDE_HIDDEN_ELEMENTS,
        mode: str = DEFAULT_RESPONSE_MODE,
    ):
        """
        Initialize the Custom Browser Tool Specification.

        Args:
            async_browser: An async Playwright browser instance.
            timeout_for_data: Seconds to wait for data extraction before timing out.
            timeout_for_element: Seconds to wait for element retrieval before timing out.
            wait_for_network_idle: Whether to wait for network idle state.
            include_hidden_for_data: Include visually hidden elements in data extraction.
            include_hidden_for_element: Include visually hidden elements in element retrieval.
            mode: Processing mode ('standard' for deep analysis, 'fast' for quick results).
        """
        self.async_browser = async_browser
        self.timeout_for_data = timeout_for_data
        self.timeout_for_element = timeout_for_element
        self.wait_for_network_idle = wait_for_network_idle
        self.include_hidden_for_data = include_hidden_for_data
        self.include_hidden_for_element = include_hidden_for_element
        self.mode = mode

    async def _get_active_page(self) -> Page:
        """
        Get the active page from the browser context.

        Returns:
            Page: The active Playwright page.
        """
        contexts = self.async_browser.contexts
        if not contexts:
            raise ValueError("No browser contexts available.")
        context = contexts[0]
        pages = context.pages
        if not pages:
            raise ValueError("No pages available in the browser context.")
        return pages[0]

    async def extract_web_data_from_browser(
        self,
        query: Optional[str] = None,
        prompt: Optional[str] = None,
        content_selectors: Optional[List[str]] = None,
        exclude_selectors: Optional[List[str]] = None,
        extra_metadata: Optional[List[str]] = None,
    ) -> dict:
        """
        Extracts structured data as JSON from a web page using a query or prompt, including body content.

        Args:
            query: A structured query to extract data (e.g., CSS selector or custom format).
            prompt: Natural language description of the data to extract.
            content_selectors: List of CSS selectors for main content (e.g., ['main', 'article']).
            exclude_selectors: List of CSS selectors to exclude (e.g., ['script', 'nav']).
            extra_metadata: List of additional metadata fields to extract (e.g., ['keywords', 'description']).

        Returns:
            dict: Extracted data including body content and metadata (e.g., title, author, content).

        Raises:
            ValueError: If neither query nor prompt is provided, or if both are provided.
        """
        if not query and not prompt:
            raise ValueError(QUERY_PROMPT_REQUIRED_ERROR_MESSAGE)
        if query and prompt:
            raise ValueError(QUERY_PROMPT_EXCLUSIVE_ERROR_MESSAGE)

        page = await self._get_active_page()

        if self.wait_for_network_idle:
            await page.wait_for_load_state("networkidle", timeout=self.timeout_for_data * 1000)

        data = {}

        # Extract body content
        try:
            content_selectors = content_selectors or DEFAULT_CONTENT_SELECTORS
            exclude_selectors = exclude_selectors or DEFAULT_EXCLUDE_SELECTORS
            content = []
            for selector in content_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    if await element.is_visible() or self.include_hidden_for_data:
                        # Exclude unwanted elements
                        for exclude in exclude_selectors:
                            excluded = await element.query_selector_all(exclude)
                            for ex in excluded:
                                await ex.evaluate("element => element.remove()")
                        text = await element.text_content()
                        if text:
                            # Clean up text (remove excessive whitespace)
                            cleaned_text = " ".join(text.strip().split())
                            if cleaned_text:
                                content.append(cleaned_text)
            data["content"] = content if content else [
                "No visible content found"]
        except Exception as e:
            data["content_error"] = f"Body content extraction failed: {str(e)}"

        if query:
            # Handle query-based extraction (e.g., CSS selector)
            try:
                elements = await page.query_selector_all(query)
                if elements:
                    texts = [await element.text_content() for element in elements if await element.is_visible() or self.include_hidden_for_data]
                    data["query_results"] = [text.strip()
                                             for text in texts if text]
                else:
                    data["query_results"] = []
            except Exception as e:
                data["query_error"] = f"Query extraction failed: {str(e)}"
        else:
            # Handle prompt-based extraction with metadata
            try:
                # Extract standard metadata
                title = await page.title()
                data["title"] = title.strip() if title else "Unknown"

                author_meta = await page.query_selector("meta[name='author']")
                data["author"] = (await author_meta.get_attribute("content")).strip() if author_meta else "Unknown"

                date_meta = await page.query_selector("meta[name='publication-date'], meta[property='article:published_time']")
                data["publication_date"] = (await date_meta.get_attribute("content")).strip() if date_meta else "Unknown"

                abstract = await page.query_selector(".abstract, [class*='abstract'], section p")
                data["abstract"] = (await abstract.text_content()).strip() if abstract and (await abstract.is_visible() or self.include_hidden_for_data) else "Unknown"

                data["url"] = page.url

                # Extract extra metadata if provided
                if extra_metadata:
                    for field in extra_metadata:
                        meta = await page.query_selector(f"meta[name='{field}'], meta[property='{field}']")
                        data[field] = (await meta.get_attribute("content")).strip() if meta else "Unknown"
            except Exception as e:
                data["prompt_error"] = f"Prompt extraction failed: {str(e)}"

        return data

    async def get_web_element_from_browser(
        self,
        prompt: str,
    ) -> str:
        """
        Finds a web element on the active page using a natural language prompt and returns its CSS selector.

        Args:
            prompt: Natural language description of the web element.

        Returns:
            str: CSS selector of the target element.

        Raises:
            ValueError: If no element is found matching the prompt.
        """
        page = await self._get_active_page()

        if self.wait_for_network_idle:
            await page.wait_for_load_state("networkidle", timeout=self.timeout_for_element * 1000)

        # Simplified element selection logic based on prompt
        # Map natural language prompt to common element types
        prompt = prompt.lower().strip()
        selector = None

        if "button" in prompt:
            selector = "button"
        elif "link" in prompt or "hyperlink" in prompt:
            selector = "a"
        elif "input" in prompt or "field" in prompt:
            selector = "input, textarea"
        elif "image" in prompt:
            selector = "img"
        else:
            selector = "*"  # Fallback to any element

        try:
            elements = await page.query_selector_all(selector)
            for element in elements:
                if await element.is_visible() or self.include_hidden_for_element:
                    # Use a unique attribute or generate a custom selector
                    element_id = await element.get_attribute("id")
                    if element_id:
                        return f"#{element_id}"
                    # Fallback to generating a unique selector based on index
                    index = elements.index(element)
                    return f"{selector}:nth-of-type({index + 1})"
            raise ValueError("No matching element found for the prompt.")
        except Exception as e:
            raise ValueError(f"Element retrieval failed: {str(e)}")
