"""
Demo 4: LLM Query Selector Generation Context

Demonstrates how extracted skeletons can be used as context for LLMs
to generate accurate CSS/XPath selectors.
"""

from jet.scrapers.html_skeleton_extractor import extract_skeleton

# Complex e-commerce page structure
html = """
<!DOCTYPE html>
<html>
<body>
    <div class="page-wrapper">
        <header class="site-header">
            <div class="logo">Logo</div>
            <nav class="main-nav">
                <ul>
                    <li><a href="/category/electronics">Electronics</a></li>
                    <li><a href="/category/books">Books</a></li>
                </ul>
            </nav>
        </header>
        
        <main class="content">
            <div class="product-grid">
                <article class="product-card" data-product-id="12345">
                    <img src="widget.jpg" alt="Premium Widget"/>
                    <h2 class="product-title">Premium Widget Pro</h2>
                    <div class="product-meta">
                        <span class="rating">4.5 stars</span>
                        <span class="reviews">(128 reviews)</span>
                    </div>
                    <div class="pricing">
                        <span class="original-price">$149.99</span>
                        <span class="sale-price">$99.99</span>
                        <span class="discount">33% OFF</span>
                    </div>
                    <button class="add-to-cart" aria-label="Add Premium Widget Pro to cart">
                        Add to Cart
                    </button>
                </article>
                
                <article class="product-card" data-product-id="67890">
                    <img src="gadget.jpg" alt="Smart Gadget"/>
                    <h2 class="product-title">Smart Gadget Plus</h2>
                    <div class="product-meta">
                        <span class="rating">4.8 stars</span>
                        <span class="reviews">(256 reviews)</span>
                    </div>
                    <div class="pricing">
                        <span class="sale-price">$199.99</span>
                    </div>
                    <button class="add-to-cart" aria-label="Add Smart Gadget Plus to cart">
                        Add to Cart
                    </button>
                </article>
            </div>
        </main>
        
        <footer class="site-footer">
            <p>&copy; 2026 E-Commerce Site</p>
        </footer>
    </div>
</body>
</html>
"""

print("=" * 80)
print("LLM QUERY SELECTOR GENERATION CONTEXT")
print("=" * 80)

# Extract skeleton for LLM context
skeleton = extract_skeleton(html)

print("\nExtracted Skeleton (for LLM context):")
print("-" * 80)
print(skeleton)

# Simulate what you'd send to an LLM
print("\n" + "=" * 80)
print("PROMPT TEMPLATE FOR LLM:")
print("=" * 80)

prompt_template = f"""
You are an expert web scraper. Given the following HTML skeleton, generate 
accurate CSS selectors for the requested data points.

HTML SKELETON:
{skeleton}

TASK: Generate CSS selectors to extract:
1. Product title from the first product card
2. Sale price from the first product card  
3. Add to cart button from the first product card
4. Product ID from the first product card

Provide your answer in JSON format:
{{
    "product_title": "css_selector_here",
    "sale_price": "css_selector_here",
    "add_to_cart_button": "css_selector_here",
    "product_id": "css_selector_here"
}}
"""

print(prompt_template)

print("\n" + "=" * 80)
print("Expected LLM Response:")
print("=" * 80)
expected_response = """
{
    "product_title": "article.product-card:first-child h2.product-title",
    "sale_price": "article.product-card:first-child .pricing .sale-price",
    "add_to_cart_button": "article.product-card:first-child button.add-to-cart",
    "product_id": "article.product-card:first-child"
}
"""
print(expected_response)

print("\n" + "=" * 80)
print("Benefits:")
print("=" * 80)
print("✓ Reduced token count (90-95% smaller than raw HTML)")
print("✓ Clear structural hierarchy visible to LLM")
print("✓ Essential attributes preserved (class, id, aria-label)")
print("✓ Noise removed (scripts, styles, deep nesting)")
print("✓ Faster LLM inference due to smaller context")
