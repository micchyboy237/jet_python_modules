# client_translate.py
from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Literal, Sequence, overload, Generator

import httpx
from rich.logging import RichHandler
from tqdm.auto import tqdm

# Configure rich logging once
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, show_time=False)],
)
log = logging.getLogger(__name__)

Device = Literal["cpu", "cuda"]

# --- Global persistent client (thread-local for safety) ---
_client_local = threading.local()

def _get_client() -> httpx.Client:
    if not getattr(_client_local, "client", None):
        _client_local.client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(120.0, connect=10.0),  # Increased timeout, faster connect fail
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )
        log.info("Created persistent httpx.Client for translation")
    return _client_local.client

@contextmanager
def _persistent_client() -> Generator[httpx.Client, None, None]:
    yield _get_client()

@overload
def translate_text(text: str, *, device: Device = "cuda", base_url: str = "http://shawn-pc.local:8001") -> dict[str, str]: ...


@overload
def translate_text(
    text: Sequence[str],
    *,
    device: Device = "cuda",
    base_url: str = "http://shawn-pc.local:8001",
    show_progress: bool = True,
) -> list[dict[str, str]]: ...


def translate_text(
    text: str | Sequence[str],
    *,
    device: Device = "cuda",
    base_url: str = "http://shawn-pc.local:8001",
    show_progress: bool = True,
) -> dict[str, str] | list[dict[str, str]]:
    """
    Translate Japanese text → English using your local Faster-Whisper API server.

    Args:
        text: Single string or list of strings to translate.
        device: ``"cuda"`` or ``"cpu"`` – passed to the backend.
        base_url: Base URL of the translation server (without trailing slash).
        show_progress: Show tqdm progress bar when translating batches.

    Returns:
        If input is str → single result dict with keys ``"original"`` and ``"translation"``.
        If input is list → list of result dicts in same order.

    Raises:
        httpx.HTTPStatusError: If the server returns an error status.
        httpx.RequestError: On network issues.
    """
    if isinstance(text, str):
        text_list: list[str] = [text]
        single_input = True
    else:
        text_list = list(text)
        single_input = False

    if not text_list:
        return [] if not single_input else {"original": "", "translation": ""} if single_input else []

    endpoint = f"{base_url.rstrip('/')}/translate"

    log.info(f"Translating {len(text_list)} text(s) → {endpoint} [device={device}]")

    # Use long-lived persistent httpx.Client via our contextmanager
    with _persistent_client() as client:
        resp = client.post(
            endpoint,
            json={"text": text_list},
            params={"device": device},
        )

    resp.raise_for_status()
    data = resp.json()

    results: list[dict[str, str]] = data.get("results", [])

    # Preserve original order and handle possible server-side mismatches
    translated = []
    iterator = tqdm(results, desc="Processing results", leave=False) if show_progress and len(results) > 5 else results

    for item in iterator:
        translated.append({"original": item.get("original", ""), "translation": item.get("translation", "")})

    return translated[0] if single_input else translated


# Convenience helper for quick interactive use
def print_translations(
    results: dict[str, str] | list[dict[str, str]],
) -> None:
    """Pretty-print translation results using rich."""
    from rich.table import Table
    from rich.console import Console

    console = Console()

    if isinstance(results, dict):
        results = [results]

    table = Table(title="Translation Results", show_header=True, header_style="bold magenta")
    table.add_column("Original", style="cyan", justify="left")
    table.add_column("Translation", style="green", justify="left")

    for r in results:
        table.add_row(r["original"], r["translation"])

    console.print(table)

    # INSERT_YOUR_CODE
if __name__ == "__main__":
    # Single
    result = translate_text("今日はいい天気ですね。一緒に散歩しませんか？")
    print_translations(result)

    # Batch
    texts = [
        "昨日、友達と一緒に映画を見に行きました。",
        "日本は美しい国ですね！",
        "今日の天気はとても良いです。",
    ]
    results = translate_text(texts, device="cuda")
    print_translations(results)
