"""
Demo 2: Custom Configuration

Demonstrates how to customize the extraction behavior using SkeletonConfig.
Useful for fine-tuning the output for specific use cases.
"""

from jet.scrapers.html_skeleton_extractor import HTMLSkeletonExtractor, SkeletonConfig

# Sample HTML
html = """
<div class="container">
    <!-- This is a comment -->
    <h1>Title</h1>
    <p>This is a very long paragraph that contains many words and should be 
    truncated based on our configuration settings to reduce token count 
    when feeding to an LLM for processing.</p>
    <div><div><span>Nested content</span></div></div>
</div>
"""

print("=" * 80)
print("CUSTOM CONFIGURATION DEMO")
print("=" * 80)

# Example 1: Preserve comments and disable merging
print("\n1. Preserving Comments & Disabling Merge:")
print("-" * 80)
config1 = SkeletonConfig(
    preserve_comments=True, merge_single_children=False, max_text_length=20
)
extractor1 = HTMLSkeletonExtractor(config1)
result1 = extractor1.extract(html)
print(result1)

# Example 2: Aggressive text truncation
print("\n2. Aggressive Text Truncation (max 5 words):")
print("-" * 80)
config2 = SkeletonConfig(max_text_length=5, text_placeholder="[...]")
extractor2 = HTMLSkeletonExtractor(config2)
result2 = extractor2.extract(html)
print(result2)

# Example 3: Custom attribute whitelist
print("\n3. Custom Attribute Whitelist (only id and data-testid):")
print("-" * 80)
config3 = SkeletonConfig(
    essential_attributes=frozenset(["id", "data-testid"]), remove_empty_elements=False
)
extractor3 = HTMLSkeletonExtractor(config3)
html_with_attrs = (
    '<div id="main" class="container" data-testid="wrapper"><p>Text</p></div>'
)
result3 = extractor3.extract(html_with_attrs)
print(result3)

print("\n" + "=" * 80)
print("Configuration Options:")
print("=" * 80)
print("• max_text_length: Control text truncation")
print("• merge_single_children: Enable/disable container merging")
print("• preserve_comments: Keep or remove HTML comments")
print("• essential_attributes: Customize which attributes to keep")
print("• remove_empty_elements: Control empty element removal")
