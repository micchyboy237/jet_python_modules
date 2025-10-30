"""
HTTPX Global Interceptor Setup for Embedding Endpoints
======================================================

Call `setup_llamacpp_embed_interceptors(base_urls=[...])` at app startup to enable
interception of embedding requests to specified base URLs (e.g., http://shawn-pc.local:8081/v1).

Works globally with:
- httpx.Client / AsyncClient
- openai.Embedding.create (used by LlamaCPP embedding clients)
- Any library using httpx

Usage:
    from jet.shared_modules.shared.setup.httpx_interceptor import setup_llamacpp_embed_interceptors
    setup_llamacpp_embed_interceptors(base_urls=["http://shawn-pc.local:8081/v1"])
"""

from __future__ import annotations

import httpx
from typing import Callable, Optional, Set
import time
from datetime import datetime
import json
from urllib.parse import urlparse

from jet.adapters.llama_cpp.tokens import count_tokens


# === DEFAULT CONFIG ===
DEFAULT_BASE_URLS = {"http://shawn-pc.local:8081/v1"}
# =====================

class EmbeddingInterceptor:
    """Intercepts embedding requests for specified base URLs."""
    
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
        """Check if URL matches any configured base_url and targets /embeddings."""
        try:
            parsed = urlparse(str(url))
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rsplit('/embeddings', 1)[0]}".rstrip("/")
            return any(base.startswith(target) for target in self.base_urls) and "/embeddings" in parsed.path
        except Exception:
            return False

    def _get_request_id(self, request: httpx.Request) -> str:
        return f"{id(request):x}"

    def _sanitize(self, data: dict | list) -> str:
        """Return a printable, truncated representation of the data."""
        return json.dumps(data, indent=2, ensure_ascii=False)[: self.max_content_length]

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
        data = json.loads(body or "{}", strict=False) if body else {}

        # Handle both list[str] and dict with "input"
        input_texts = data.get("input") or data
        if isinstance(input_texts, str):
            input_texts = [input_texts]

        input_tokens = sum(count_tokens([text], data.get("model", "")) for text in input_texts)

        msg = (
            f"\n{'='*80}\n"
            f"EMBEDDING REQUEST → {request.method} {request.url}\n"
            f"ID: {rid} | {datetime.now():%H:%M:%S.%f}[:-3]\n"
            f"HEADERS:\n{self._format_headers(request.headers)}\n"
            f"PARAMS: {dict(request.url.params)}\n"
            f"TOKENS: {json.dumps({'input': input_tokens}, indent=2)}\n"
            f"BODY:\n```json\n{self._sanitize(data) if body else 'None'}\n```\n"
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

        content = response.text
        data = json.loads(content or "{}", strict=False)

        # Count tokens in usage.prompt_tokens if available, fallback to estimating from input
        prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
        embedding_count = len(data.get("data", []))

        msg = (
            f"\n{'='*80}\n"
            f"EMBEDDING RESPONSE ← {response.status_code} {response.url}\n"
            f"ID: {rid} | {duration:.1f}ms\n"
            f"HEADERS:\n{self._format_headers(response.headers)}\n"
            f"EMBEDDINGS RETURNED: {embedding_count}\n"
            f"TOKENS (usage): {json.dumps({'prompt_tokens': prompt_tokens}, indent=2)}\n"
            f"BODY:\n```json\n{self._sanitize(data) if content else 'No content'}\n```\n"
            f"{'='*80}"
        )
        self.logger(msg)


# === GLOBAL STATE ===
_patched = False
_original_client_init = None
_original_async_client_init = None
_interceptor: Optional[EmbeddingInterceptor] = None


def _patch_client_init():
    global _original_client_init, _interceptor
    if _original_client_init is None:
        _original_client_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        _original_client_init(self, *args, **kwargs)
        base_url = kwargs.get("base_url") or getattr(self, "base_url", "")
        if _interceptor and _interceptor._should_intercept(httpx.URL(base_url)):
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
        base_url = kwargs.get("base_url") or getattr(self, "base_url", "")
        if _interceptor and _interceptor._should_intercept(httpx.URL(base_url)):
            hooks = getattr(self, "event_hooks", {}) or {}
            hooks.setdefault("request", []).append(_interceptor.request_hook)
            hooks.setdefault("response", []).append(_interceptor.response_hook)
            self.event_hooks = hooks

    httpx.AsyncClient.__init__ = patched_init  # type: ignore


def setup_logger():
    import os
    from jet.llm.config import DEFAULT_LOG_DIR
    from jet.logger import logger, CustomLogger

    embedding_log_file = f"{DEFAULT_LOG_DIR}/embedding.log"

    os.makedirs(os.path.dirname(embedding_log_file), exist_ok=True)
    if os.path.exists(embedding_log_file):
        os.remove(embedding_log_file)

    embedding_logger = CustomLogger("embedding", filename=embedding_log_file)
    logger.orange(f"Embedding REST logs: {embedding_log_file}")
    return embedding_logger


def setup_llamacpp_embed_interceptors(
    base_urls: Optional[set[str] | list[str]] = None,
    *,
    logger: Optional[Callable[[str], None]] = None,
    include_sensitive: bool = False,
    max_content_length: int = 2000,
    force: bool = False,
) -> None:
    """
    Enable global interception for embedding endpoints.

    Args:
        base_urls: Set or list of base URLs to intercept (e.g., ["http://api.local:8081/v1"]).
                   Defaults to ["http://shawn-pc.local:8081/v1"].
        logger: Custom logger function (defaults to file-based).
        include_sensitive: Log sensitive headers (default: False).
        max_content_length: Truncate body preview length.
        force: Re-apply patch even if already set up.
    """
    global _patched, _interceptor

    if _patched and not force:
        return

    if base_urls is None:
        base_urls = DEFAULT_BASE_URLS
    elif isinstance(base_urls, list):
        base_urls = set(base_urls)
    elif not isinstance(base_urls, set):
        raise ValueError("base_urls must be a set or list of strings")

    for url in base_urls:
        try:
            urlparse(url)
        except Exception as e:
            raise ValueError(f"Invalid base_url: {url} ({e})")

    if not logger:
        logger = setup_logger()

    _interceptor = EmbeddingInterceptor(
        base_urls=base_urls,
        logger=logger,
        include_sensitive=include_sensitive,
        max_content_length=max_content_length,
    )

    _patch_client_init()
    _patch_async_client_init()
    _patched = True

    print(f"\nHTTPX Embedding Interceptors ENABLED for {', '.join(base_urls)}")


# === Optional: Manual client factories ===
def create_client(**kwargs) -> httpx.Client:
    setup_llamacpp_embed_interceptors()
    return httpx.Client(**kwargs)


def create_async_client(**kwargs) -> httpx.AsyncClient:
    setup_llamacpp_embed_interceptors()
    return httpx.AsyncClient(**kwargs)


# === Demo ===
def _run_demo():
    setup_llamacpp_embed_interceptors(base_urls=["http://shawn-pc.local:8081/v1"])
    print("\nDemo: Intercepted embedding call")
    with httpx.Client(base_url="http://shawn-pc.local:8081/v1") as c:
        c.post("/embeddings", json={"input": ["Hello world"], "model": "nomic-embed-text"})
    print("\nDemo: Ignored non-embedding call")
    with httpx.Client(base_url="http://shawn-pc.local:8081/v1") as c:
        c.get("/models")

if __name__ == "__main__":
    _run_demo()