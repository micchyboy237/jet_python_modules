"""
HTTPX Global Interceptor Setup
=============================

Call `setup_llamacpp_interceptors(base_urls=[...])` at app startup to enable
interception of all requests to specified base URLs (e.g., http://shawn-pc.local:8080/v1).

Works globally with:
- httpx.Client / AsyncClient
- openai.Client (used by ChatLlamaCpp)
- Any library using httpx

Usage:
    from jet.shared_modules.shared.setup.httpx_interceptor import setup_llamacpp_interceptors
    setup_llamacpp_interceptors(base_urls=["http://shawn-pc.local:8080/v1"])
"""

from __future__ import annotations

import httpx
from typing import Callable, Optional, Set
import time
from datetime import datetime
import json
from urllib.parse import urlparse

# === DEFAULT CONFIG ===
DEFAULT_BASE_URLS = {"http://shawn-pc.local:8080/v1"}
# =====================

class LocalInterceptor:
    """Intercepts requests for specified base URLs."""
    
    def __init__(
        self,
        base_urls: Set[str],
        logger: Optional[Callable[[str], None]] = None,
        include_sensitive: bool = False,
        max_content_length: int = 2000,
    ):
        self.base_urls = {url.rstrip("/") for url in base_urls}
        self.logger = logger or print
        self.include_sensitive = include_sensitive
        self.max_content_length = max_content_length
        self.start_times: dict[str, float] = {}

    def _should_intercept(self, url: str) -> bool:
        """Check if URL matches any configured base_url."""
        try:
            parsed = urlparse(str(url))
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            return any(base.startswith(target) for target in self.base_urls)
        except Exception:
            return False

    def _get_request_id(self, request: httpx.Request) -> str:
        return f"{id(request):x}"

    def _sanitize(self, content: str) -> str:
        if len(content) > self.max_content_length:
            return content[:self.max_content_length] + f"... [TRUNCATED {len(content)}]"
        return content

    def _format_headers(self, headers: httpx.Headers) -> str:
        h = dict(headers)
        if not self.include_sensitive:
            for k in list(h):
                if any(s in k.lower() for s in ["authorization", "cookie", "token", "x-api-key"]):
                    h[k] = "***HIDDEN***"
        return json.dumps(h, indent=2)

    def request_hook(self, request: httpx.Request) -> None:
        if not self._should_intercept(request.url):
            return
        rid = self._get_request_id(request)
        self.start_times[rid] = time.time()

        body = request.content.decode("utf-8", errors="ignore") if request.content else None

        msg = (
            f"\n{'='*80}\n"
            f"REQUEST → {request.method} {request.url}\n"
            f"ID: {rid} | {datetime.now():%H:%M:%S.%f}[:-3]\n"
            f"HEADERS:\n{self._format_headers(request.headers)}\n"
            f"PARAMS: {dict(request.url.params)}\n"
            f"BODY: {self._sanitize(body) if body else 'None'}\n"
            f"{'='*80}"
        )
        self.logger(msg)

    def response_hook(self, response: httpx.Response) -> None:
        if not self._should_intercept(response.url):
            return
        rid = self._get_request_id(response.request)
        duration = (time.time() - self.start_times.pop(rid, time.time())) * 1000

        try:
            if not response.is_closed:
                response.read()
        except Exception:
            pass

        content = response.text or "No content"

        msg = (
            f"\n{'='*80}\n"
            f"RESPONSE ← {response.status_code} {response.url}\n"
            f"ID: {rid} | {duration:.1f}ms\n"
            f"HEADERS:\n{self._format_headers(response.headers)}\n"
            f"BODY: {self._sanitize(content)}\n"
            f"{'='*80}"
        )
        self.logger(msg)


# === GLOBAL STATE ===
_patched = False
_original_client_init = None
_original_async_client_init = None
_interceptor: Optional[LocalInterceptor] = None


def _patch_client_init():
    global _original_client_init, _interceptor
    if _original_client_init is None:
        _original_client_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        _original_client_init(self, *args, **kwargs)
        if _interceptor and _interceptor._should_intercept(kwargs.get("base_url", getattr(self, "base_url", ""))):
            hooks = getattr(self, "event_hooks", {}) or {}
            hooks.setdefault("request", []).append(_interceptor.request_hook)
            hooks.setdefault("response", []).append(_interceptor.response_hook)
            self.event_hooks = hooks

    httpx.Client.__init__ = patched_init  # type: ignore


def _patch_async_client_init():
    global _original_async_client_init, _interceptor
    if _original_async_client_init is None:
        _original_async_client_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        _original_async_client_init(self, *args, **kwargs)
        if _interceptor and _interceptor._should_intercept(kwargs.get("base_url", getattr(self, "base_url", ""))):
            hooks = getattr(self, "event_hooks", {}) or {}
            hooks.setdefault("request", []).append(_interceptor.request_hook)
            hooks.setdefault("response", []).append(_interceptor.response_hook)
            self.event_hooks = hooks

    httpx.AsyncClient.__init__ = patched_init  # type: ignore


def setup_llamacpp_interceptors(
    base_urls: Optional[set[str] | list[str]] = None,
    *,
    logger: Optional[Callable[[str], None]] = None,
    include_sensitive: bool = False,
    max_content_length: int = 2000,
    force: bool = False,
) -> None:
    """
    Enable global interception for specified base URLs.

    Args:
        base_urls: Set or list of base URLs to intercept (e.g., ["http://api.local:8080/v1"]).
                   Defaults to ["http://shawn-pc.local:8080/v1"].
        logger: Custom logger function (defaults to print).
        include_sensitive: Log sensitive headers (default: False).
        max_content_length: Truncate body preview length.
        force: Re-apply patch even if already set up.
    """
    global _patched, _interceptor

    if _patched and not force:
        return

    # Normalize base_urls to a set
    if base_urls is None:
        base_urls = DEFAULT_BASE_URLS
    elif isinstance(base_urls, list):
        base_urls = set(base_urls)
    elif not isinstance(base_urls, set):
        raise ValueError("base_urls must be a set or list of strings")

    # Validate URLs
    for url in base_urls:
        try:
            urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid base_url: {url} ({e})")

    # Create interceptor
    _interceptor = LocalInterceptor(
        base_urls=base_urls,
        logger=logger,
        include_sensitive=include_sensitive,
        max_content_length=max_content_length,
    )

    # Apply patches
    _patch_client_init()
    _patch_async_client_init()
    _patched = True

    print(f"\nHTTPX Interceptors ENABLED for {', '.join(base_urls)}")


# === Optional: Manual client factories ===
def create_client(**kwargs) -> httpx.Client:
    setup_llamacpp_interceptors()
    return httpx.Client(**kwargs)


def create_async_client(**kwargs) -> httpx.AsyncClient:
    setup_llamacpp_interceptors()
    return httpx.AsyncClient(**kwargs)


# === Demo ===
def _run_demo():
    setup_llamacpp_interceptors(base_urls=["http://shawn-pc.local:8080/v1", "http://api.local:9090"])
    print("\nDemo: Intercepted calls")
    with httpx.Client(base_url="http://shawn-pc.local:8080/v1") as c:
        c.get("/models")
    with httpx.Client(base_url="http://api.local:9090") as c:
        c.get("/test")
    print("\nDemo: Ignored call")
    with httpx.Client(base_url="https://httpbin.org") as c:
        c.get("/get")

if __name__ == "__main__":
    _run_demo()