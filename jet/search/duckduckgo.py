from typing import TypedDict, Optional, Dict, Any, Literal
from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException
from jet.logger import logger

# Type definitions based on sample results


class TextResult(TypedDict):
    title: str
    href: str
    body: str


class NewsResult(TypedDict):
    date: str
    title: str
    body: str
    url: str
    image: str
    source: str


class ImageResult(TypedDict):
    title: str
    image: str
    thumbnail: str
    url: str
    height: int
    width: int
    source: str


class VideoImages(TypedDict):
    large: str
    medium: str
    motion: str
    small: str


class VideoStatistics(TypedDict):
    viewCount: int


class VideoResult(TypedDict):
    title: str
    content: Optional[str]
    description: str
    duration: str
    embed_html: str
    embed_url: Optional[str]
    image_token: str
    images: VideoImages
    provider: str
    published: str
    publisher: str
    statistics: VideoStatistics
    uploader: str


class BookResult(TypedDict):
    title: str
    url: Optional[str]
    href: Optional[str]
    content: Optional[str]


class DuckDuckGoSearch:
    """A typed wrapper for the DDGS search client."""

    def __init__(
        self,
        proxy: Optional[str] = None,
        timeout: int = 5,
        verify: bool = True
    ):
        """Initialize the DDGS client with optional proxy, timeout, and SSL verification."""
        self.ddgs = DDGS(proxy=proxy, timeout=timeout, verify=verify)

    def __enter__(self) -> "DuckDuckGoSearch":
        """Enter context manager."""
        self.ddgs.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        self.ddgs.__exit__(exc_type, exc_val, exc_tb)

    def text(
        self,
        query: str,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: Optional[str] = None,
        max_results: Optional[int] = 10,
        page: int = 1,
        backend: str = "auto"
    ) -> list[TextResult]:
        """Perform a text search."""
        try:
            results = self.ddgs.text(
                query=query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                max_results=max_results,
                page=page,
                backend=backend
            )
            return [TextResult(**result) for result in results]
        except DDGSException as e:
            logger.error(f"Text search error: {e}")
            return []

    def news(
        self,
        query: str,
        timelimit: Optional[str] = None,
        max_results: Optional[int] = 10,
        backend: str = "auto"
    ) -> list[NewsResult]:
        """Perform a news search."""
        try:
            results = self.ddgs.news(
                query=query,
                timelimit=timelimit,
                max_results=max_results,
                backend=backend
            )
            return [NewsResult(**result) for result in results]
        except DDGSException as e:
            logger.error(f"News search error: {e}")
            return []

    def images(
        self,
        query: str,
        region: str = "us-en",
        safesearch: str = "moderate",
        max_results: Optional[int] = 10,
        backend: str = "auto"
    ) -> list[ImageResult]:
        """Perform an image search."""
        try:
            results = self.ddgs.images(
                query=query,
                region=region,
                safesearch=safesearch,
                max_results=max_results,
                backend=backend
            )
            return [ImageResult(**result) for result in results]
        except DDGSException as e:
            logger.error(f"Image search error: {e}")
            return []

    def videos(
        self,
        query: str,
        max_results: Optional[int] = 10,
        backend: str = "auto"
    ) -> list[VideoResult]:
        """Perform a video search."""
        try:
            results = self.ddgs.videos(
                query=query,
                max_results=max_results,
                backend=backend
            )
            return [VideoResult(**result) for result in results]
        except DDGSException as e:
            logger.error(f"Video search error: {e}")
            return []

    def books(
        self,
        query: str,
        max_results: Optional[int] = 10,
        backend: str = "auto"
    ) -> list[BookResult]:
        """Perform a book search."""
        try:
            results = self.ddgs.books(
                query=query,
                max_results=max_results,
                backend=backend
            )
            return [BookResult(**result) for result in results]
        except DDGSException as e:
            logger.error(f"Book search error: {e}")
            return []
