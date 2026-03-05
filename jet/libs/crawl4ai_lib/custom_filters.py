# custom_filters.py
from typing import Optional
from urllib.parse import urlparse

from crawl4ai.deep_crawling.filters import URLFilter


class MaxPathSegmentsFilter(URLFilter):
    """
    Rejects URLs whose path has more segments than allowed.

    Examples:
        max_segments=3 allows:
        /a/b/c           → 3 segments
        /blog/2025/jan   → 3 segments
        /docs/           → 1 segment
        /                → 0 segments

        Rejects /a/b/c/d → 4 segments
    """

    __slots__ = ("max_segments", "_cache")

    def __init__(self, max_segments: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"MaxPathSegmentsFilter({max_segments})")
        if max_segments < 0:
            raise ValueError("max_segments must be >= 0")
        self.max_segments = max_segments
        self._cache = {}  # very simple manual cache (or use @lru_cache)

    def apply(self, url: str) -> bool:
        # Quick bypass for very common cases
        if (
            self.max_segments >= 8
        ):  # arbitrary threshold — most sites don't go this deep
            self._update_stats(True)
            return True

        key = url
        if key in self._cache:
            result = self._cache[key]
            self._update_stats(result)
            return result

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")  # normalize trailing slash

        if not path or path == "/":
            count = 0
        else:
            count = path.count(
                "/"
            )  # number of segments = count("/") after stripping trailing /

        result = count <= self.max_segments
        self._cache[key] = result
        self._update_stats(result)
        return result
