"""
E-commerce Product Scraping Example

This example demonstrates scraping product pages and
extracting structured information for market research.
"""


from swarms_tools.search.web_scraper import (
    scrape_single_url_sync,
    format_scraped_content,
)
from jet.file.utils import save_file
from jet.logger import CustomLogger
from datetime import datetime
import os
import shutil

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
log_dir = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)
filename_no_ext = os.path.splitext(os.path.basename(__file__))[0]
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

# Scrape product information from e-commerce sites
product_urls = [
    "https://www.amazon.com/dp/B08N5WRWNW",  # Example product
    "https://store.steampowered.com",  # Digital products
]

# Configure for e-commerce scraping
ecommerce_content = scrape_single_url_sync(
    url=product_urls[1],  # Use Steam store as example
    timeout=15,
    user_agent="Mozilla/5.0 (compatible; PriceBot/1.0)",
    remove_scripts=True,
    remove_styles=True,
)

# Extract product information
product_name = ecommerce_content.title
product_description = ecommerce_content.text
product_images = ecommerce_content.images
related_links = ecommerce_content.links

# Format for summary view
product_summary = format_scraped_content(
    ecommerce_content, format_type="summary", truncate=True
)
logger.info("Product Summary:\n%s", product_summary)

# Format for detailed analysis
product_details = format_scraped_content(
    ecommerce_content, format_type="detailed", truncate=False
)
logger.info("Product Details:\n%s", product_details)

# Compute updated analysis metrics
page_url = ecommerce_content.url if hasattr(ecommerce_content, "url") else product_urls[1]
page_title = product_name
title_length = len(product_name) if product_name else 0
word_count = len(product_description.split()) if product_description else 0
link_count = len(related_links) if related_links else 0
image_count = len(product_images) if product_images else 0
text_length = len(product_description) if product_description else 0
avg_word_length = (text_length / word_count) if word_count else 0
has_title = bool(product_name and product_name.strip())
content_richness = (word_count / link_count) if link_count else 0
media_ratio = (image_count / word_count * 100) if word_count else 0

# Log updated analysis metrics
logger.info(
    "E-commerce Product Analysis for %s:\n"
    "  Product Title: %s\n"
    "  Title Length: %d chars\n"
    "  Description Length: %d chars\n"
    "  Word Count: %d\n"
    "  Number of Images: %d\n"
    "  Number of Links: %d\n"
    "  Average Word Length: %.2f chars\n"
    "  Has Title: %s\n"
    "  Content Richness (words/link): %.2f\n"
    "  Media Ratio (images/words): %.2f%%\n"
    "  Images: %s\n"
    "  Related Links: %s",
    page_url,
    page_title,
    title_length,
    text_length,
    word_count,
    image_count,
    link_count,
    avg_word_length,
    has_title,
    content_richness,
    media_ratio,
    product_images,
    related_links
)

save_file(product_summary, f"{OUTPUT_DIR}/product_summary.md")
save_file(product_details, f"{OUTPUT_DIR}/product_details.md")
save_file(ecommerce_content, f"{OUTPUT_DIR}/ecommerce_content.json")

# Use cases:
# - Price monitoring
# - Product research
# - Inventory tracking
# - Market analysis
# - Competitor monitoring
