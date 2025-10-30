"""
HTTPX Global Interceptor Setup
=============================

Call `setup_httpx_interceptors()` once at app startup to enable
interception of all requests to:

    http://shawn-pc.local:8080/v1/*

Works globally with:
- httpx.Client / AsyncClient
- openai.Client (used by ChatLlamaCpp)
- Any library using httpx

Usage:
    from jet.shared_modules.shared.setup import httpx_interceptor
    httpx_interceptor.setup_httpx_interceptors()
"""

from __future__ import annotations

import httpx
from typing import Callable, Optional
import time
from datetime import datetime
import json
from urllib.parse import urlparse

# === CONFIG ===
TARGET_BASE = "http://shawn-pc.local:8080/v1"
# =============

class LocalV1Interceptor:
    """Intercepts only requests under TARGET_BASE."""
    
    def __init__(
        self,
        logger: Optional[Callable[[str], None]] = None,
        include_sensitive: bool = False,
        max_content_length: int = 2000,
    ):
        self.logger = logger or print
        self.include_sensitive = include_sensitive
        self.max_content_length = max_content_length
        self.start_times: dict[str, float] = {}

    def _should_intercept(self, url: str) -> bool:
        try:
            parsed = urlparse(str(url))
            base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            return base.startswith(TARGET_BASE.rstrip("/"))
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


def _patch_client_init():
    global _original_client_init
    if _original_client_init is None:
        _original_client_init = httpx.Client.__init__

    def patched_init(self, *args, **kwargs):
        _original_client_init(self, *args, **kwargs)
        base_url = kwargs.get("base_url") or getattr(self, "base_url", None)
        if base_url and str(base_url).startswith(TARGET_BASE):
            interceptor = LocalV1Interceptor()
            hooks = getattr(self, "event_hooks", {}) or {}
            hooks.setdefault("request", []).append(interceptor.request_hook)
            hooks.setdefault("response", []).append(interceptor.response_hook)
            self.event_hooks = hooks

    httpx.Client.__init__ = patched_init  # type: ignore


def _patch_async_client_init():
    global _original_async_client_init
    if _original_async_client_init is None:
        _original_async_client_init = httpx.AsyncClient.__init__

    def patched_init(self, *args, **kwargs):
        _original_async_client_init(self, *args, **kwargs)
        base_url = kwargs.get("base_url") or getattr(self, "base_url", None)
        if base_url and str(base_url).startswith(TARGET_BASE):
            interceptor = LocalV1Interceptor()
            hooks = getattr(self, "event_hooks", {}) or {}
            hooks.setdefault("request", []).append(interceptor.request_hook)
            hooks.setdefault("response", []).append(interceptor.response_hook)
            self.event_hooks = hooks

    httpx.AsyncClient.__init__ = patched_init  # type: ignore


def setup_httpx_interceptors(
    *,
    logger: Optional[Callable[[str], None]] = None,
    include_sensitive: bool = False,
    max_content_length: int = 2000,
    force: bool = False,
) -> None:
    """
    Enable global interception for http://shawn-pc.local:8080/v1

    Args:
        logger: Custom logger (defaults to print)
        include_sensitive: Show auth headers (default: False)
        max_content_length: Truncate body preview
        force: Re-apply even if already patched
    """
    global _patched

    if _patched and not force:
        return

    # Override default logger in interceptor
    if logger:
        LocalV1Interceptor.__init__.__defaults__ = (logger, include_sensitive, max_content_length)  # type: ignore

    _patch_client_init()
    _patch_async_client_init()
    _patched = True

    print(f"HTTPX Interceptors ENABLED for {TARGET_BASE}/*")


# === Optional: Manual client factories ===
def create_client(**kwargs) -> httpx.Client:
    setup_httpx_interceptors()
    return httpx.Client(**kwargs)

def create_async_client(**kwargs) -> httpx.AsyncClient:
    setup_httpx_interceptors()
    return httpx.AsyncClient(**kwargs)


# === Demo ===
def _run_demo():
    setup_httpx_interceptors()
    print("\nDemo: Intercepted call")
    with httpx.Client(base_url=TARGET_BASE) as c:
        c.get("/models")

if __name__ == "__main__":
    _run_demo()