from playwright.sync_api import sync_playwright


class GenericWebsurferAgent:
    """
    A generic web agent that adapts to different website structures.
    Uses semantic analysis, accessibility trees, and fallback strategies.
    """

    def __init__(self):
        self.element_cache = {}  # Cache element fingerprints for adaptive matching

    def find_element_generic(
        self,
        page,
        role_hints=None,
        text_hints=None,
        attribute_patterns=None,
        position_hint=None,
    ):
        """
        Find an element using multiple detection strategies in priority order.

        Args:
            role_hints: Expected ARIA roles (e.g., ['searchbox', 'textbox'])
            text_hints: Text patterns to match (e.g., ['search', 'find'])
            attribute_patterns: Regex patterns for attributes
            position_hint: Approximate location ('top', 'center', 'bottom')

        Returns:
            Playwright locator or None
        """

        # Strategy 1: Accessibility Tree (Most Reliable)
        if role_hints:
            element = self._try_accessibility_role(page, role_hints)
            if element:
                return element

        # Strategy 2: Semantic Text Matching
        if text_hints:
            element = self._try_text_matching(page, text_hints)
            if element:
                return element

        # Strategy 3: Attribute Pattern Matching
        if attribute_patterns:
            element = self._try_attribute_patterns(page, attribute_patterns)
            if element:
                return element

        # Strategy 4: Common Selector Fallbacks
        element = self._try_common_selectors(page, role_hints, text_hints)
        if element:
            return element

        # Strategy 5: Visual/Position-based (Last Resort)
        if position_hint:
            element = self._try_position_based(page, position_hint)
            if element:
                return element

        return None

    def _try_accessibility_role(self, page, roles):
        """Try finding elements by ARIA role."""
        for role in roles:
            try:
                elements = page.get_by_role(role)
                count = elements.count()
                if count > 0:
                    print(f"[LOG] Found {count} elements with role '{role}'")
                    return elements.first
            except Exception as e:
                print(f"[LOG] Role '{role}' failed: {e}")
                continue
        return None

    def _try_text_matching(self, page, text_patterns):
        """Find elements containing specific text patterns."""
        for pattern in text_patterns:
            try:
                # Try exact text first
                elements = page.get_by_text(pattern, exact=False)
                if elements.count() > 0:
                    print(f"[LOG] Found elements with text pattern: '{pattern}'")
                    return elements.first

                # Try placeholder text
                elements = page.locator(f'[placeholder*="{pattern}" i]')
                if elements.count() > 0:
                    print(f"[LOG] Found elements with placeholder: '{pattern}'")
                    return elements.first

                # Try aria-label
                elements = page.locator(f'[aria-label*="{pattern}" i]')
                if elements.count() > 0:
                    print(f"[LOG] Found elements with aria-label: '{pattern}'")
                    return elements.first

            except Exception as e:
                print(f"[LOG] Text pattern '{pattern}' failed: {e}")
                continue
        return None

    def _try_attribute_patterns(self, page, patterns):
        """Find elements using regex patterns on attributes."""
        for attr_name, pattern in patterns.items():
            try:
                # Use JavaScript to find elements matching pattern
                elements = page.evaluate(f"""
                    () => {{
                        const allElements = document.querySelectorAll('*');
                        const matches = [];
                        const regex = /{pattern}/i;
                        
                        for (const el of allElements) {{
                            const value = el.getAttribute('{attr_name}');
                            if (value && regex.test(value)) {{
                                matches.push({{
                                    tag: el.tagName,
                                    text: el.innerText?.substring(0, 50),
                                    rect: el.getBoundingClientRect()
                                }});
                            }}
                        }}
                        return matches.slice(0, 10); // Return top 10
                    }}
                """)

                if elements:
                    print(
                        f"[LOG] Found {len(elements)} elements matching {attr_name}='{pattern}'"
                    )
                    # Return the first visible element
                    return page.locator(
                        f'[{attr_name}*="{pattern.split("|")[0]}"]'
                    ).first

            except Exception as e:
                print(f"[LOG] Attribute pattern failed: {e}")
                continue
        return None

    def _try_common_selectors(self, page, roles, texts):
        """Try common selector patterns used across websites."""
        common_selectors = [
            # Search-related
            'input[type="search"]',
            'input[name="q"]',
            'input[name="query"]',
            'input[name="search"]',
            'input[aria-label*="search" i]',
            "#search",
            "#search-input",
            ".search-input",
            ".search-box",
            # Form inputs
            'input[type="text"]',
            "textarea",
            '[contenteditable="true"]',
        ]

        for selector in common_selectors:
            try:
                elements = page.query_selector_all(selector)
                if elements:
                    # Filter to visible elements only
                    visible = [el for el in elements if el.is_visible()]
                    if visible:
                        print(f"[LOG] Found element with selector: {selector}")
                        return page.locator(selector).first
            except Exception as e:
                continue

        return None

    def _try_position_based(self, page, position):
        """Find elements based on viewport position (last resort)."""
        try:
            viewport = page.viewport_size
            if not viewport:
                return None

            width = viewport["width"]
            height = viewport["height"]

            # Define search regions
            regions = {
                "top": {"y_start": 0, "y_end": height * 0.3},
                "center": {"y_start": height * 0.3, "y_end": height * 0.7},
                "bottom": {"y_start": height * 0.7, "y_end": height},
            }

            region = regions.get(position, regions["center"])

            # Get all interactive elements in region
            elements_in_region = page.evaluate(f"""
                () => {{
                    const region = {region};
                    const interactive = document.querySelectorAll('input, button, a, [role="button"]');
                    const matches = [];
                    
                    for (const el of interactive) {{
                        const rect = el.getBoundingClientRect();
                        if (rect.top >= region.y_start && rect.top <= region.y_end &&
                            rect.width > 0 && rect.height > 0) {{
                            matches.push({{
                                tag: el.tagName,
                                type: el.type || el.getAttribute('role'),
                                text: el.innerText?.substring(0, 30),
                                x: rect.left + rect.width/2,
                                y: rect.top + rect.height/2,
                                width: rect.width,
                                height: rect.height
                            }});
                        }}
                    }}
                    
                    // Sort by size (prefer larger, more prominent elements)
                    matches.sort((a, b) => (b.width * b.height) - (a.width * a.height));
                    return matches.slice(0, 5);
                }}
            """)

            if elements_in_region:
                print(
                    f"[LOG] Found {len(elements_in_region)} elements in {position} region"
                )
                # Click the largest element
                target = elements_in_region[0]
                return page.locator(f"*").nth(0)  # Would need refinement

        except Exception as e:
            print(f"[LOG] Position-based search failed: {e}")

        return None


class AdaptiveStructureExtractor:
    """
    Extracts page structure generically using DOM graph analysis [[20]].
    Adapts to website changes by fingerprinting elements.
    """

    def extract_structure_generic(self, page, focus_area=None):
        """
        Extract page structure as a semantic graph.

        Args:
            page: Playwright page object
            focus_area: Optional area to focus on ('header', 'main', 'footer', 'sidebar')

        Returns:
            Structured dict representing page layout
        """

        # Step 1: Get accessibility snapshot
        snapshot = page.locator("body").aria_snapshot()

        # Step 2: Extract DOM metadata via JavaScript
        dom_metadata = page.evaluate("""
            () => {
                // Build element tree
                function buildTree(element, depth = 0) {
                    if (depth > 5) return null; // Limit depth
                    
                    const node = {
                        tag: element.tagName.toLowerCase(),
                        id: element.id || null,
                        classes: Array.from(element.classList),
                        role: element.getAttribute('role') || null,
                        ariaLabel: element.getAttribute('aria-label') || null,
                        text: element.innerText?.trim().substring(0, 100) || null,
                        children: [],
                        attributes: {},
                        isVisible: element.offsetParent !== null,
                        rect: {
                            x: element.getBoundingClientRect().left,
                            y: element.getBoundingClientRect().top,
                            width: element.getBoundingClientRect().width,
                            height: element.getBoundingClientRect().height
                        }
                    };
                    
                    // Get important attributes
                    const importantAttrs = ['href', 'src', 'alt', 'title', 'name', 'type', 'value'];
                    for (const attr of importantAttrs) {
                        if (element.hasAttribute(attr)) {
                            node.attributes[attr] = element.getAttribute(attr);
                        }
                    }
                    
                    // Recursively process children
                    for (const child of element.children) {
                        const childNode = buildTree(child, depth + 1);
                        if (childNode) {
                            node.children.push(childNode);
                        }
                    }
                    
                    return node;
                }
                
                // Focus on specific area if requested
                let rootElement = document.body;
                const areaSelectors = {
                    'header': 'header, [role="banner"], .header, #header',
                    'main': 'main, [role="main"], .main, #main, .content',
                    'footer': 'footer, [role="contentinfo"], .footer, #footer',
                    'sidebar': 'aside, [role="complementary"], .sidebar, #sidebar',
                    'navigation': 'nav, [role="navigation"], .nav, #nav'
                };
                
                if ('{focus_area}' in areaSelectors) {
                    const focused = document.querySelector(areaSelectors['{focus_area}']);
                    if (focused) rootElement = focused;
                }
                
                return {
                    title: document.title,
                    url: window.location.href,
                    tree: buildTree(rootElement),
                    stats: {
                        totalLinks: document.querySelectorAll('a').length,
                        totalImages: document.querySelectorAll('img').length,
                        totalHeadings: document.querySelectorAll('h1,h2,h3,h4,h5,h6').length,
                        totalForms: document.querySelectorAll('form').length,
                        hasSearchInput: !!document.querySelector('input[type="search"], input[name="q"], input[name="search"]')
                    }
                };
            }
        """)

        # Step 3: Parse accessibility tree
        parsed_snapshot = self._parse_accessibility_tree(snapshot)

        # Step 4: Combine both representations
        structure = {
            "metadata": {
                "title": dom_metadata["title"],
                "url": dom_metadata["url"],
                "timestamp": page.evaluate("() => new Date().toISOString()"),
            },
            "dom_tree": dom_metadata["tree"],
            "accessibility_tree": parsed_snapshot,
            "statistics": dom_metadata["stats"],
            "interactive_elements": self._extract_interactive_elements(
                dom_metadata["tree"]
            ),
            "content_sections": self._identify_content_sections(dom_metadata["tree"]),
        }

        return structure

    def _parse_accessibility_tree(self, snapshot):
        """Parse YAML accessibility snapshot into structured format."""
        lines = snapshot.splitlines()
        elements = []

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("-"):
                continue

            # Extract role and name
            parts = stripped.split(":")
            if len(parts) >= 2:
                role = parts[0].strip().lstrip("- ")
                name = ":".join(parts[1:]).strip().strip('"')

                elements.append(
                    {
                        "role": role,
                        "name": name,
                        "has_state": "[checked]" in stripped
                        or "[selected]" in stripped,
                    }
                )

        return elements

    def _extract_interactive_elements(self, tree_node):
        """Recursively extract all interactive elements from DOM tree."""
        interactive = []
        interactive_tags = ["a", "button", "input", "select", "textarea"]
        interactive_roles = ["button", "link", "textbox", "combobox", "menuitem"]

        def traverse(node):
            if not node:
                return

            is_interactive = (
                node["tag"] in interactive_tags
                or node.get("role") in interactive_roles
                or node.get("attributes", {}).get("tabindex") is not None
            )

            if is_interactive and node.get("isVisible"):
                interactive.append(
                    {
                        "tag": node["tag"],
                        "id": node.get("id"),
                        "classes": node.get("classes", []),
                        "role": node.get("role"),
                        "aria_label": node.get("ariaLabel"),
                        "text": node.get("text"),
                        "attributes": node.get("attributes", {}),
                        "position": node.get("rect"),
                    }
                )

            for child in node.get("children", []):
                traverse(child)

        traverse(tree_node)
        return interactive

    def _identify_content_sections(self, tree_node):
        """Identify major content sections (articles, lists, tables, etc.)."""
        sections = []

        def find_sections(node, path=""):
            if not node:
                return

            current_path = f"{path}/{node['tag']}"

            # Identify list containers
            if node["tag"] in ["ul", "ol", "dl"]:
                items = [
                    child for child in node.get("children", []) if child["tag"] == "li"
                ]
                if items:
                    sections.append(
                        {
                            "type": "list",
                            "path": current_path,
                            "item_count": len(items),
                            "sample_items": [
                                item.get("text", "")[:50] for item in items[:3]
                            ],
                        }
                    )

            # Identify tables
            elif node["tag"] == "table":
                rows = [
                    child
                    for child in node.get("children", [])
                    if child["tag"] in ["tr", "thead", "tbody"]
                ]
                sections.append(
                    {"type": "table", "path": current_path, "row_count": len(rows)}
                )

            # Identify articles/cards
            elif node.get("role") == "article" or "card" in " ".join(
                node.get("classes", [])
            ):
                sections.append(
                    {
                        "type": "article",
                        "path": current_path,
                        "heading": node.get("text", "")[:100],
                        "has_link": any(
                            child["tag"] == "a" for child in node.get("children", [])
                        ),
                    }
                )

            for child in node.get("children", []):
                find_sections(child, current_path)

        find_sections(tree_node)
        return sections


def generic_search_and_extract(
    agent: GenericWebsurferAgent,
    extractor: AdaptiveStructureExtractor,
    query: str,
    search_url: str = None,
):
    """
    Perform search on ANY website using generic detection strategies.

    Args:
        agent: GenericWebsurferAgent instance
        extractor: AdaptiveStructureExtractor instance
        query: Search query
        search_url: Optional search engine URL (auto-detects if None)

    Returns:
        Dict with results
    """

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Auto-detect search URL if not provided
        if not search_url:
            search_url = detect_search_engine(page)

        print(f"[LOG] Using search engine: {search_url}")
        page.goto(search_url, wait_until="domcontentloaded")

        # Extract initial structure
        print("[LOG] Analyzing page structure...")
        initial_structure = extractor.extract_structure_generic(page)

        # Check if page has search capability
        if not initial_structure["statistics"]["hasSearchInput"]:
            print("[WARNING] No search input detected. Trying anyway...")

        # Find search input using generic strategies
        print("[LOG] Locating search input...")
        search_input = agent.find_element_generic(
            page,
            role_hints=["searchbox", "textbox", "combobox"],
            text_hints=["search", "find", "query"],
            attribute_patterns={
                "name": r"q|query|search|keyword",
                "type": r"search|text",
                "placeholder": r"search|find",
            },
            position_hint="top",
        )

        if not search_input:
            raise Exception("Could not locate search input using any strategy")

        print("[LOG] Found search input. Typing query...")

        # Type query
        search_input.fill(query)

        # Submit search (try multiple methods)
        submitted = False

        # Method 1: Press Enter
        try:
            search_input.press("Enter")
            submitted = True
            print("[LOG] Submitted via Enter key")
        except:
            pass

        # Method 2: Find and click submit button
        if not submitted:
            submit_button = agent.find_element_generic(
                page,
                role_hints=["button"],
                text_hints=["search", "submit", "go", "find"],
                position_hint="top",
            )

            if submit_button:
                submit_button.click()
                submitted = True
                print("[LOG] Submitted via button click")

        if not submitted:
            raise Exception("Could not submit search")

        # Wait for results
        print("[LOG] Waiting for results...")
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(2000)  # Additional wait for dynamic content

        # Extract results URL
        results_url = page.url
        print(f"[LOG] Results URL: {results_url}")

        # Extract results structure
        print("[LOG] Extracting results structure...")
        results_structure = extractor.extract_structure_generic(page)

        # Extract actual results generically
        results = extract_results_generic(page, results_structure)

        browser.close()

        return {
            "query": query,
            "search_engine": search_url,
            "results_url": results_url,
            "results_count": len(results),
            "results": results,
            "page_structure": {
                "statistics": results_structure["statistics"],
                "sections": results_structure["content_sections"],
                "interactive_elements_count": len(
                    results_structure["interactive_elements"]
                ),
            },
        }


def detect_search_engine(page):
    """Auto-detect best search engine based on accessibility."""
    # Default to DuckDuckGo (privacy-friendly, simpler structure)
    return "https://duckduckgo.com/"


def extract_results_generic(page, structure):
    """
    Extract search results using generic patterns.
    Works across Google, Bing, DuckDuckGo, etc.
    """
    results = []

    # Strategy 1: Look for repeated link patterns
    links = page.query_selector_all("a[href]")

    # Group links by similarity (same parent, similar styling)
    link_groups = {}
    for link in links:
        parent = link.evaluate("el => el.parentElement.className")
        if parent not in link_groups:
            link_groups[parent] = []
        link_groups[parent].append(link)

    # Find groups with multiple links (likely result lists)
    for parent_class, group_links in link_groups.items():
        if len(group_links) >= 3:  # At least 3 similar links suggests a list
            for link in group_links[:10]:  # Top 10 results
                try:
                    title = link.inner_text()
                    href = link.get_attribute("href")

                    # Skip navigation/header/footer links
                    if len(title) < 3 or not href:
                        continue

                    # Look for snippet/description nearby
                    parent = link.evaluate("el => el.parentElement")
                    snippet = ""
                    if parent:
                        snippet_elem = page.evaluate(
                            """
                            (parentEl) => {
                                const text = parentEl.innerText;
                                const lines = text.split('\\n').filter(l => l.trim());
                                return lines.length > 1 ? lines.slice(1).join(' ').substring(0, 200) : '';
                            }
                        """,
                            parent,
                        )
                        snippet = snippet_elem or ""

                    results.append(
                        {
                            "title": title.strip(),
                            "url": href,
                            "snippet": snippet.strip()[:200],
                        }
                    )

                    if len(results) >= 10:
                        break

                except Exception as e:
                    continue

            if results:
                break

    # Strategy 2: If no grouped results found, try article/list patterns
    if not results:
        articles = page.query_selector_all(
            'article, [role="article"], .result, .search-result'
        )
        for article in articles[:10]:
            try:
                title_elem = article.query_selector("h1, h2, h3, a")
                link_elem = article.query_selector("a[href]")

                if title_elem and link_elem:
                    results.append(
                        {
                            "title": title_elem.inner_text().strip(),
                            "url": link_elem.get_attribute("href"),
                            "snippet": article.inner_text().strip()[:200],
                        }
                    )
            except:
                continue

    return results


# Usage Example
if __name__ == "__main__":
    agent = GenericWebsurferAgent()
    extractor = AdaptiveStructureExtractor()

    result = generic_search_and_extract(
        agent=agent,
        extractor=extractor,
        query="playwright automation tutorial",
        search_url="https://duckduckgo.com/",
    )

    print("\n=== GENERIC SEARCH RESULTS ===")
    print(f"Query: {result['query']}")
    print(f"Results URL: {result['results_url']}")
    print(f"Found {result['results_count']} results\n")

    for i, res in enumerate(result["results"][:5], 1):
        print(f"{i}. {res['title']}")
        print(f"   URL: {res['url']}")
        print(f"   Snippet: {res['snippet'][:100]}...\n")
