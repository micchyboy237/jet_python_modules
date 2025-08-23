from jet.search.duckduckgo import DuckDuckGoSearch, TextResult, NewsResult, ImageResult, VideoResult, BookResult
from jet.logger import logger
from ddgs.exceptions import DDGSException, TimeoutException


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
    if props:
        props.pop()
        colors.pop()
    return props, colors


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


def example_news_custom_date_range():
    try:
        with DuckDuckGoSearch() as client:
            # Try both date range formats to identify the correct one
            for date_format, timelimit in [
                ("colon", "2025-08-01:2025-12-31"),
                ("dots", "2025-08-01..2025-12-31")
            ]:
                logger.info(
                    f"\nTesting news search with {date_format} format: {timelimit}")
                results: list[NewsResult] = client.news(
                    query="artificial intelligence",
                    timelimit=timelimit,
                    max_results=10
                )
                for num, result in enumerate(results, start=1):
                    props, colors = format_result_properties(result)
                    logger.log(
                        f"News {num} (Custom Date Range 2025, {date_format} format)",
                        *props,
                        colors=["DEBUG", *colors],
                    )
    except DDGSException as e:
        logger.error(f"Error: {e}")


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
    logger.info("\nExample 4: News Search with Custom Date Range")
    example_news_custom_date_range()
    logger.info("\nExample 5: Video Search with Default Backend")
    example_video_search()
    logger.info("\nExample 6: Book Search without Proxy")
    example_book_search()
    logger.info("\nExample 7: Handling Timeout Exception")
    example_timeout_handling()
    logger.info("\nExample 8: Custom Page and Max Results")
    example_custom_page()
    logger.info("\nExample 9: Multiple Backends")
    example_multiple_backends()
    logger.info("\nExample 10: Disabling SSL Verification")
    example_disable_ssl_verification()
