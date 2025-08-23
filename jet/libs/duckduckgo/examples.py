import logging
from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

from jet.logger import logger

# Configure logging for better debugging
# logging.basicConfig(level=logging.INFO)


# Example 1: Basic Text Search
# Demonstrates a simple text search with default parameters
def example_basic_text_search():
    try:
        with DDGS() as ddgs:
            results = ddgs.text("python programming")
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Result {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['href'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 2: Image Search with Custom Parameters
# Shows how to perform an image search with specific region and safesearch settings
def example_image_search():
    try:
        with DDGS(timeout=10) as ddgs:
            results = ddgs.images(
                query="cute cats",
                region="uk-en",
                safesearch="off",
                max_results=5
            )
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Image {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['image'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 3: News Search with Time Limit
# Demonstrates news search with a timelimit to get recent articles
def example_news_search():
    try:
        with DDGS() as ddgs:
            results = ddgs.news(
                query="artificial intelligence",
                timelimit="d",  # Last day
                max_results=3
            )
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"News {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nSource: ", result['source'],
                    " | ",
                    "\nDate: ", result['date'],
                    colors=["DEBUG", "WHITE", "SUCCESS", "GRAY",
                            "WHITE", "SUCCESS", "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 4: Video Search with Default Backend
# Shows how to perform a video search with the default backend and handle result keys
def example_video_search():
    try:
        with DDGS() as ddgs:
            results = ddgs.videos(
                query="machine learning tutorials",
                backend="auto",  # Use default backend since 'youtube' is unavailable
                max_results=4
            )
            for num, result in enumerate(results, start=1):
                # Use 'embed_url' or 'content' for video URLs, depending on backend
                video_url = result.get('embed_url') or result.get(
                    'content') or 'No URL available'
                logger.log(
                    f"Video {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", video_url,
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 5: Book Search with Proxy
# Demonstrates book search using a proxy for network requests
def example_book_search():
    try:
        with DDGS(proxy="http://proxy.example.com:8080") as ddgs:
            results = ddgs.books(
                query="python books",
                max_results=3
            )
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Book {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['url'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 6: Handling Timeout Exception
# Shows how to handle a potential timeout during a search
def example_timeout_handling():
    try:
        with DDGS(timeout=1) as ddgs:  # Short timeout to simulate failure
            results = ddgs.text(
                "complex query with many results", max_results=100)
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Result {num}",
                    "\nData: ", str(result),
                    colors=["DEBUG", "WHITE", "SUCCESS"],
                )
    except TimeoutException as e:
        logger.error(f"Timeout Error: {e}")
    except DDGSException as e:
        logger.error(f"General Error: {e}")


# Example 7: Custom Page and Max Results
# Demonstrates fetching a specific page of results with a limited number of results
def example_custom_page():
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query="data science",
                page=2,
                max_results=5
            )
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Result {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['href'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 8: Multiple Backends
# Shows how to specify multiple backends for a search
def example_multiple_backends():
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                query="blockchain technology",
                backend="wikipedia,google",
                max_results=6
            )
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Result {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['href'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Example 9: Disabling SSL Verification
# Demonstrates search with SSL verification disabled (not recommended for production)
def example_disable_ssl_verification():
    try:
        with DDGS(verify=False) as ddgs:
            results = ddgs.text("python tutorials", max_results=3)
            for num, result in enumerate(results, start=1):
                logger.log(
                    f"Result {num}",
                    "\nTitle: ", result['title'],
                    " | ",
                    "\nURL: ", result['href'],
                    colors=["DEBUG", "WHITE", "SUCCESS",
                            "GRAY", "WHITE", "SUCCESS"],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


# Run all examples
if __name__ == "__main__":
    logger.info("\nExample 1: Basic Text Search")
    example_basic_text_search()
    logger.info("\nExample 2: Image Search with Custom Parameters")
    example_image_search()
    logger.info("\nExample 3: News Search with Time Limit")
    example_news_search()
    logger.info("\nExample 4: Video Search with Specific Backend")
    example_video_search()
    logger.info("\nExample 5: Book Search with Proxy")
    example_book_search()
    logger.info("\nExample 6: Handling Timeout Exception")
    example_timeout_handling()
    logger.info("\nExample 7: Custom Page and Max Results")
    example_custom_page()
    logger.info("\nExample 8: Multiple Backends")
    example_multiple_backends()
    logger.info("\nExample 9: Disabling SSL Verification")
    example_disable_ssl_verification()
