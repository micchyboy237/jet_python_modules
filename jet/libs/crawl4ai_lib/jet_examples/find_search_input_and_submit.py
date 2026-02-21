import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig


async def perform_search(
    url: str,
    query: str = "twerk",
    wait_for_results_selector: str = ".message, .search-results, .results, main, .thread-list, .content",
) -> None:
    """
    Attempts to find and use a search bar in a generic way.

    Priority order for finding input:
    1. type="search"
    2. placeholder containing "search" (case insensitive)
    3. name/id/class containing "search", "q", "query", "keyword"
    4. first visible text input in a form (fallback)
    """
    js_find_and_search = f"""
    (async () => {{
        // Helper: dispatch input/change events
        function fillInput(el, value) {{
            el.value = value;
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            el.focus();
        }}

        // 1. Try most semantic: type="search"
        let input = document.querySelector('input[type="search"]');

        // 2. Placeholder containing "search"
        if (!input) {{
            input = Array.from(document.querySelectorAll('input[placeholder]'))
                .find(el => el.placeholder.toLowerCase().includes('search'));
        }}

        // 3. Common names/ids/classes
        if (!input) {{
            const selectors = [
                'input[name="q"]',
                'input#search',
                'input[name="search"]',
                'input[name="query"]',
                'input[name="keyword"]',
                'input[name="s"]',           // xenforo/common
                'input#QuickSearchQuery',    // xenforo quick search
                '[class*="search"] input[type="text"]',
                '[id*="search"] input[type="text"]'
            ];
            for (const sel of selectors) {{
                input = document.querySelector(sel);
                if (input) break;
            }}
        }}

        // 4. Last resort: first visible text input inside a form
        if (!input) {{
            input = document.querySelector('form input[type="text"], form input:not([type])');
        }}

        if (!input) {{
            console.warn("No search input found");
            return;
        }}

        // Fill
        fillInput(input, "{query}");

        // Submit strategies (try in order)
        let submitted = false;

        // A. Form submit (most reliable)
        const form = input.closest('form');
        if (form) {{
            form.submit();
            submitted = true;
        }}

        // B. Enter key (many JS-handled searches)
        if (!submitted) {{
            input.dispatchEvent(new KeyboardEvent('keydown', {{
                key: 'Enter',
                code: 'Enter',
                keyCode: 13,
                bubbles: true
            }}));
            submitted = true;
        }}

        // C. Find and click submit button
        if (!submitted) {{
            const btnSelectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button[class*="search"]',
                '[aria-label*="search" i]',
                '[class*="magnifying"], .fa-search, .icon--search'
            ];
            for (const sel of btnSelectors) {{
                const btn = document.querySelector(sel);
                if (btn) {{
                    btn.click();
                    submitted = true;
                    break;
                }}
            }}
        }}

        // Small safety delay
        await new Promise(r => setTimeout(r, 1200));
    }})();
    """

    browser_conf = BrowserConfig(
        headless=True,
        java_script_enabled=True,
        verbose=True,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    run_conf = CrawlerRunConfig(
        js_code=js_find_and_search,
        wait_for=f"css:{wait_for_results_selector}",
        page_timeout=60000,  # 60 seconds total for load + wait_for
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_conf) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_conf,
        )

        if result.success:
            print(f"[SUCCESS] Crawled after searching '{query}'")
            print(f"Markdown length: {len(result.markdown or '')}")
            print(f"HTML length   : {len(result.cleaned_html or '')}")
            # Optional: print(result.markdown[:600] + "...")
        else:
            print("[FAIL]", result.error_message)


if __name__ == "__main__":
    asyncio.run(
        perform_search(
            url="https://nsfwph.org",
            query="twerk",
            # Adjust this selector based on actual results container after search
            wait_for_results_selector=".message-cell, .structItem, .searchResults, .content",
        )
    )
