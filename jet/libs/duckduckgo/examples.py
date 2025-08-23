from jet.search.duckduckgo import DuckDuckGoSearch, TextResult, NewsResult, ImageResult, VideoResult, BookResult
from jet.logger import logger
from ddgs.exceptions import DDGSException, TimeoutException

# Helper function to format result properties for logging


def format_result_properties(result: dict) -> tuple[list[str], list[str]]:
    """Format all result properties as strings with corresponding colors for logging."""
    props = []
    colors = []
    logger.newline()
    for key, value in result.items():
        props.extend([f"\n{key}: ", str(value)])
        colors.extend(["WHITE", "SUCCESS"])
        props.append(" | ")
        colors.append("GRAY")
    # Remove trailing separator
    if props:
        props.pop()
        colors.pop()
    return props, colors

# Example 1: Basic Text Search
# Demonstrates a simple text search with all result properties


def example_basic_text_search():
    try:
        with DuckDuckGoSearch() as client:
            results: list[TextResult] = client.text("python programming")
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Result {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 2: Image Search with Custom Parameters
# Shows an image search with all result properties


def example_image_search():
    try:
        with DuckDuckGoSearch(timeout=10) as client:
            results: list[ImageResult] = client.images(
                query="cute cats",
                region="uk-en",
                safesearch="off",
                max_results=5
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Image {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 3: News Search with Time Limit
# Demonstrates news search with all result properties


def example_news_search():
    try:
        with DuckDuckGoSearch() as client:
            results: list[NewsResult] = client.news(
                query="artificial intelligence",
                timelimit="d",
                max_results=3
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"News {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 4: Video Search with Default Backend
# Shows a video search with all result properties


def example_video_search():
    try:
        with DuckDuckGoSearch() as client:
            results: list[VideoResult] = client.videos(
                query="machine learning tutorials",
                backend="auto",
                max_results=4
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Video {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 5: Book Search without Proxy
# Demonstrates book search with all result properties


def example_book_search():
    try:
        with DuckDuckGoSearch() as client:
            results: list[BookResult] = client.books(
                query="python books",
                max_results=3
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Book {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 6: Handling Timeout Exception
# Shows handling a timeout with all result properties


def example_timeout_handling():
    try:
        with DuckDuckGoSearch(timeout=1) as client:
            results: list[TextResult] = client.text(
                query="complex query with many results",
                max_results=100
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Result {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except TimeoutException as e:
        logger.error(f"Timeout Error: {e}")
    except DDGSException as e:
        logger.error(f"General Error: {e}")

# Example 7: Custom Page and Max Results
# Demonstrates fetching a specific page with all result properties


def example_custom_page():
    try:
        with DuckDuckGoSearch() as client:
            results: list[TextResult] = client.text(
                query="data science",
                page=2,
                max_results=5
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Result {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 8: Multiple Backends
# Shows a search with multiple backends and all result properties


def example_multiple_backends():
    try:
        with DuckDuckGoSearch() as client:
            results: list[TextResult] = client.text(
                query="blockchain technology",
                backend="wikipedia,google",
                max_results=6
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Result {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")

# Example 9: Disabling SSL Verification
# Demonstrates search with SSL verification disabled and all result properties


def example_disable_ssl_verification():
    try:
        with DuckDuckGoSearch(verify=False) as client:
            results: list[TextResult] = client.text(
                query="python tutorials",
                max_results=3
            )
            for num, result in enumerate(results, start=1):
                props, colors = format_result_properties(result)
                logger.log(
                    f"Result {num}",
                    *props,
                    colors=["DEBUG", *colors],
                )
    except DDGSException as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    logger.info("\nExample 1: Basic Text Search")
    example_basic_text_search()
    logger.info("\nExample 2: Image Search with Custom Parameters")
    example_image_search()
    logger.info("\nExample 3: News Search with Time Limit")
    example_news_search()
    logger.info("\nExample 4: Video Search with Default Backend")
    example_video_search()
    logger.info("\nExample 5: Book Search without Proxy")
    example_book_search()
    logger.info("\nExample 6: Handling Timeout Exception")
    example_timeout_handling()
    logger.info("\nExample 7: Custom Page and Max Results")
    example_custom_page()
    logger.info("\nExample 8: Multiple Backends")
    example_multiple_backends()
    logger.info("\nExample 9: Disabling SSL Verification")
    example_disable_ssl_verification()
