import json
from typing import List
from jet.file.utils import save_file
from jet.scrapers.text_nodes import extract_text_nodes, BaseNode


def main() -> None:
    """
    Demonstrates usage of extract_text_nodes with a sample HTML string.
    Prints the extracted text nodes in a formatted JSON-like output.
    """
    # Sample HTML with headers, paragraphs, and excludable tags
    sample_html = """
    <div id="main">
        <h1 id="header">Welcome</h1>
        <p id="intro">Hello, world!</p>
        <div>
            <p id="nested">Nested content</p>
        </div>
        <nav>Navigation content</nav>
        <script>console.log('test');</script>
    </div>
    """

    # Extract text nodes with default excludes and timeout
    nodes: List[BaseNode] = extract_text_nodes(
        source=sample_html,
        excludes=["nav", "footer", "script", "style"],
        timeout_ms=1000
    )

    # Print results in a formatted way
    print("Extracted Text Nodes:")
    for node in nodes:
        node_dict = {
            "tag": node.tag,
            "text": node.text,
            "depth": node.depth,
            "raw_depth": node.raw_depth,
            "id": node.id,
            "class_names": node.class_names,
            "line": node.line,
            "html": node.get_html()
        }
        print(json.dumps(node_dict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
