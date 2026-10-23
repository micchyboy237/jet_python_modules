"""
Demo 3: Block Tree Construction

Shows how to build and analyze hierarchical block trees from HTML.
Useful for understanding document structure and relevance-based pruning.
"""

import json

from jet.scrapers.html_skeleton_extractor import BlockTreeBuilder, build_block_tree

# Sample HTML with multiple sections
html = """
<html>
<body>
    <header>
        <h1>Blog Homepage</h1>
        <nav>Navigation menu</nav>
    </header>
    <main>
        <article>
            <h2>Python Tutorial</h2>
            <p>Learn Python programming from scratch with this comprehensive guide.</p>
        </article>
        <article>
            <h2>JavaScript Guide</h2>
            <p>Master JavaScript development with modern best practices.</p>
        </article>
        <aside>
            <h3>Related Links</h3>
            <ul>
                <li>Web Development</li>
                <li>Programming Tips</li>
            </ul>
        </aside>
    </main>
    <footer>
        <p>Copyright 2026</p>
    </footer>
</body>
</html>
"""

print("=" * 80)
print("BLOCK TREE CONSTRUCTION")
print("=" * 80)

# Build the block tree
tree = build_block_tree(html, max_words=50)

print("\nBlock Tree Structure (formatted JSON):")
print("-" * 80)
print(
    json.dumps(tree, indent=2)[:2000] + "..."
    if len(json.dumps(tree, indent=2)) > 2000
    else json.dumps(tree, indent=2)
)

# Demonstrate relevance-based pruning
print("\n" + "=" * 80)
print("RELEVANCE-BASED PRUNING")
print("=" * 80)

builder = BlockTreeBuilder(max_words_per_block=50)

# Prune for Python-related content
print("\nPruning for query: 'Python'")
print("-" * 80)
pruned_python = builder.prune_by_relevance(tree, "Python")
print(
    json.dumps(pruned_python, indent=2)[:1500] + "..."
    if len(json.dumps(pruned_python, indent=2)) > 1500
    else json.dumps(pruned_python, indent=2)
)

# Prune for JavaScript-related content
print("\nPruning for query: 'JavaScript'")
print("-" * 80)
pruned_js = builder.prune_by_relevance(tree, "JavaScript")
print(
    json.dumps(pruned_js, indent=2)[:1500] + "..."
    if len(json.dumps(pruned_js, indent=2)) > 1500
    else json.dumps(pruned_js, indent=2)
)

print("\n" + "=" * 80)
print("Use Cases:")
print("=" * 80)
print("• Understanding document hierarchy")
print("• Identifying relevant content sections")
print("• Reducing context size for LLM queries")
print("• Building smart content extractors")
