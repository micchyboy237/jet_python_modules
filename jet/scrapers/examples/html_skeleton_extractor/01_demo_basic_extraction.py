"""
Demo 1: Basic HTML Skeleton Extraction

Shows the simplest way to extract a skeletal structure from HTML.
Perfect for getting started with the library.
"""

from jet.scrapers.html_skeleton_extractor import extract_skeleton

# Sample HTML with various elements
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Product Page</title>
    <script>console.log('analytics');</script>
    <style>.price { color: red; }</style>
</head>
<body>
    <header>
        <nav id="main-nav">
            <a href="/">Home</a>
            <a href="/products">Products</a>
        </nav>
    </header>
    <main>
        <article class="product">
            <h1>Premium Widget</h1>
            <p>This is an amazing product with great features.</p>
            <div class="price">$99.99</div>
            <button id="buy-btn">Buy Now</button>
        </article>
    </main>
</body>
</html>
"""

print("=" * 80)
print("BASIC HTML SKELETON EXTRACTION")
print("=" * 80)
print("\nOriginal HTML (truncated):")
print(html)

# Extract skeleton
skeleton = extract_skeleton(html)

print("\n" + "-" * 80)
print("Extracted Skeleton:")
print("-" * 80)
print(skeleton)

print("\n" + "=" * 80)
print("Key Observations:")
print("=" * 80)
print("✓ Script and style tags removed")
print("✓ Semantic structure preserved (header, nav, main, article)")
print("✓ Essential attributes kept (id, class)")
print("✓ Text content truncated to placeholders")
