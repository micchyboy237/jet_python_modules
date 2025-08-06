import json
from typing import List
from jet.scrapers.header_hierarchy import extract_header_hierarchy, HeaderDoc


def main() -> None:
    """
    Demonstrates usage of extract_header_hierarchy with a sample HTML string.
    Prints the extracted header sections in a formatted JSON-like output.
    """
    # Sample HTML with headers, nested elements, and excludable tags
    sample_html = """
    <p>Intro content to be ignored.</p>
    <h1>Main Header</h1>
    <p>Main content.</p>
    <div>
        <h2>Nested Sub Header</h2>
        <p>Sub content.</p>
    </div>
    <nav>Navigation content</nav>
    <script>console.log('test');</script>
    """

    # Extract header hierarchy with default excludes and timeout
    headings: List[HeaderDoc] = extract_header_hierarchy(
        source=sample_html,
        excludes=["nav", "footer", "script", "style"],
        timeout_ms=1000
    )

    # Print results in a formatted way
    print("Extracted Header Sections:")
    for heading in headings:
        print(json.dumps(heading, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
