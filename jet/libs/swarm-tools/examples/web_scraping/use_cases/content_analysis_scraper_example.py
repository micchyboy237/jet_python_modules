"""
Content Analysis Scraping Example

This example demonstrates scraping for content analysis,
SEO research, and text mining applications.
"""

import os
from datetime import datetime
from swarms_tools.search.web_scraper import SuperFastScraper
from jet.logger import CustomLogger

# Configure logging
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logger = CustomLogger(log_file, overwrite=True)

# Create analyzer-optimized scraper
logger.info("Initializing SuperFastScraper with optimized settings")
content_analyzer = SuperFastScraper(
    timeout=12,
    max_workers=4,
    user_agent="ContentAnalyzer/1.0 (compatible)",
    strip_html=True,
    remove_scripts=True,
    remove_styles=True,
    remove_comments=True,
)

# Target URLs for content analysis
analysis_targets = [
    # "docs.llamaindex.ai",
    "docs.llamaindex.ai/en/stable/examples/memory/ChatSummaryMemoryBuffer",
]

# Scrape for detailed content analysis
logger.info("Starting scrape for %d target URLs", len(analysis_targets))
try:
    analysis_results = content_analyzer.scrape_urls(analysis_targets)
    logger.info("Successfully scraped %d URLs", len(analysis_results))
except Exception as e:
    logger.error("Error during scraping: %s", str(e))
    raise

# Process each result for analysis metrics
for content in analysis_results:
    try:
        page_url = content.url
        page_title = content.title
        content_text = content.text
        word_density = content.word_count
        link_count = len(content.links)
        image_count = len(content.images)

        # Calculate content metrics
        title_length = len(content.title)
        text_length = len(content.text)
        avg_word_length = text_length / max(content.word_count, 1)

        # SEO analysis data points
        has_title = len(content.title) > 0
        content_richness = content.word_count / max(link_count, 1)
        media_ratio = image_count / max(content.word_count, 1) * 100

        # Log analysis metrics
        logger.info(
            "Analysis for %s:\n"
            "  Title: %s (Length: %d chars)\n"
            "  Word Count: %d\n"
            "  Link Count: %d\n"
            "  Image Count: %d\n"
            "  Avg Word Length: %.2f chars\n"
            "  Has Title: %s\n"
            "  Content Richness: %.2f\n"
            "  Media Ratio: %.2f%%",
            page_url,
            page_title,
            title_length,
            word_density,
            link_count,
            image_count,
            avg_word_length,
            has_title,
            content_richness,
            media_ratio
        )
    except Exception as e:
        logger.error("Error processing content for %s: %s", content.url, str(e))

# Applications:
# - SEO content audits
# - Competitive content analysis
# - Text mining and NLP preprocessing
# - Content quality assessment
# - Topic modeling preparation