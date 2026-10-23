"""
Demo 5: Real-World Web Scraping Workflow

Shows a complete workflow from fetching a webpage to extracting structured data
using skeleton extraction and LLM-powered selector generation.
"""

from jet.scrapers.html_skeleton_extractor import build_block_tree, extract_skeleton

# Simulated scraped HTML from a news website
news_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Tech News Today - Latest Updates</title>
    <script async src="https://www.googletagmanager.com/gtag/js"></script>
    <link rel="stylesheet" href="/css/main.css">
</head>
<body>
    <div id="app">
        <header role="banner">
            <div class="header-container">
                <h1 class="site-title">Tech News Today</h1>
                <nav aria-label="Main navigation">
                    <ul>
                        <li><a href="/tech">Technology</a></li>
                        <li><a href="/science">Science</a></li>
                        <li><a href="/business">Business</a></li>
                    </ul>
                </nav>
            </div>
        </header>
        
        <main role="main">
            <section class="featured-articles">
                <article class="article-card featured" data-article-id="art-001">
                    <figure>
                        <img src="/images/ai-breakthrough.jpg" alt="AI Breakthrough Image"/>
                        <figcaption>Researchers achieve new AI milestone</figcaption>
                    </figure>
                    <h2 class="article-title">Breakthrough in Artificial Intelligence Research</h2>
                    <div class="article-meta">
                        <time datetime="2026-10-23">October 23, 2026</time>
                        <span class="author">By Dr. Sarah Chen</span>
                        <span class="read-time">5 min read</span>
                    </div>
                    <p class="article-excerpt">
                        Scientists at leading research institutions have announced a major 
                        breakthrough in artificial intelligence that could revolutionize how 
                        machines understand and process human language...
                    </p>
                    <a href="/articles/ai-breakthrough" class="read-more">Read Full Article</a>
                </article>
                
                <article class="article-card" data-article-id="art-002">
                    <figure>
                        <img src="/images/quantum-computing.jpg" alt="Quantum Computer"/>
                    </figure>
                    <h2 class="article-title">Quantum Computing Reaches New Milestone</h2>
                    <div class="article-meta">
                        <time datetime="2026-10-22">October 22, 2026</time>
                        <span class="author">By James Wilson</span>
                    </div>
                    <p class="article-excerpt">
                        A team of physicists has successfully demonstrated quantum supremacy 
                        with a new 1000-qubit processor, marking a significant step forward...
                    </p>
                    <a href="/articles/quantum-milestone" class="read-more">Read More</a>
                </article>
            </section>
            
            <aside class="sidebar">
                <div class="trending-topics">
                    <h3>Trending Topics</h3>
                    <ul>
                        <li><a href="/tag/ai">#ArtificialIntelligence</a></li>
                        <li><a href="/tag/quantum">#QuantumComputing</a></li>
                        <li><a href="/tag/startups">#TechStartups</a></li>
                    </ul>
                </div>
            </aside>
        </main>
        
        <footer role="contentinfo">
            <p>&copy; 2026 Tech News Today. All rights reserved.</p>
        </footer>
    </div>
    
    <script src="/js/analytics.js"></script>
    <script src="/js/app.bundle.js"></script>
</body>
</html>
"""

print("=" * 80)
print("REAL-WORLD WEB SCRAPING WORKFLOW")
print("=" * 80)

# Step 1: Extract skeleton
print("\nStep 1: Extract HTML Skeleton")
print("-" * 80)
skeleton = extract_skeleton(news_html)
print(f"Skeleton length: {len(skeleton)} characters")
print(f"Original HTML length: {len(news_html)} characters")
print(f"Reduction: {((len(news_html) - len(skeleton)) / len(news_html) * 100):.1f}%")
print("\nSkeleton preview:")
print(skeleton[:500] + "...")

# Step 2: Build block tree for analysis
print("\n\nStep 2: Build Block Tree for Content Analysis")
print("-" * 80)
tree = build_block_tree(news_html, max_words=100)


def count_articles(node, count=0):
    """Count article nodes in the tree."""
    if node.get("tag") == "article":
        count += 1
    for child in node.get("children", []):
        count = count_articles(child, count)
    return count


article_count = count_articles(tree)
print(f"Found {article_count} article elements in the tree")

# Step 3: Simulate LLM-powered extraction
print("\n\nStep 3: LLM-Powered Data Extraction Prompt")
print("-" * 80)

extraction_prompt = f"""
Extract structured data from this news website skeleton.

SKELETON:
{skeleton[:2000]}

Extract the following for EACH article:
- Title
- Author
- Date
- Excerpt (first 100 chars)
- Article ID
- Read more link

Return as JSON array.
"""

print(extraction_prompt)

# Step 4: Show expected structured output
print("\n\nStep 4: Expected Structured Output")
print("-" * 80)
import json

expected_output = [
    {
        "title": "Breakthrough in Artificial Intelligence Research",
        "author": "Dr. Sarah Chen",
        "date": "2026-10-23",
        "excerpt": "Scientists at leading research institutions have announced...",
        "article_id": "art-001",
        "read_more_link": "/articles/ai-breakthrough",
    },
    {
        "title": "Quantum Computing Reaches New Milestone",
        "author": "James Wilson",
        "date": "2026-10-22",
        "excerpt": "A team of physicists has successfully demonstrated quantum...",
        "article_id": "art-002",
        "read_more_link": "/articles/quantum-milestone",
    },
]

print(json.dumps(expected_output, indent=2))

print("\n" + "=" * 80)
print("Workflow Benefits:")
print("=" * 80)
print("✓ 90%+ reduction in input tokens for LLM")
print("✓ Preserved semantic structure for accurate extraction")
print("✓ Removed tracking scripts and analytics code")
print("✓ Maintained essential metadata (ARIA labels, data attributes)")
print("✓ Faster processing and lower API costs")
print("✓ More reliable extraction across different page layouts")
