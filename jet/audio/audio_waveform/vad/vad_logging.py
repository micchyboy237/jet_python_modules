# live_vad_segments.py

"""
Live VAD Segmenter — streams mic audio via sounddevice, accumulates speech
chunks, and fires on_segment() when a natural boundary is detected.
Optionally saves results to disk in the same layout as vad_firered_hybrid.py.

Segment-end triggers (in priority order):
  1. "silence"       — prob stayed below threshold for >= min_silence_sec
  2. "soft_silence"  — past soft_limit AND currently in silence
  3. "valley"        — past soft_limit AND a clean valley trough was found
  4. "hard_limit"    — past hard_limit (safety fallback, no trough needed)
"""

from __future__ import annotations

import time

# ── project imports ────────────────────────────────────────────────────────────
from rich.console import Console

console = Console()


# ── logging helpers ────────────────────────────────────────────────────────────

_LAST_LOG: dict[str, float] = {}


def throttled(key: str, interval_s: float = 2.0) -> bool:
    """Return True if we should emit a log for *key* (rate-limited)."""
    now = time.monotonic()
    if now - _LAST_LOG.get(key, 0.0) >= interval_s:
        _LAST_LOG[key] = now
        return True
    return False


def log_listening() -> None:
    if throttled("idle", 15.0):
        console.print("[dim cyan]🎙  Listening…[/dim cyan]")


def log_speech_start(prob: float) -> None:
    console.print(
        f"[bold green]▶  Speech started[/bold green]  [dim](prob={prob:.2f})[/dim]"
    )


def log_accumulating(duration_s: float, prob: float) -> None:
    if throttled("accum", 1.5):
        console.print(
            f"[cyan]   ↳ accumulating[/cyan]  "
            f"[yellow]{duration_s:.1f}s[/yellow]  "
            f"[dim]prob={prob:.2f}[/dim]"
        )


def log_soft_limit_check(duration_s: float, soft_limit_s: float) -> None:
    if throttled("soft", 2.0):
        console.print(
            f"[magenta]   ⚠  Past soft limit[/magenta]  "
            f"[yellow]{duration_s:.1f}s[/yellow] > "
            f"[yellow]{soft_limit_s:.1f}s[/yellow]  — watching for valley…"
        )


def log_segment_end(reason: str, duration_s: float, extra: str = "") -> None:
    color = {
        "silence": "green",
        "soft_silence": "yellow",
        "valley": "magenta",
        "hard_limit": "red",
    }.get(reason, "white")
    console.print(
        f"[bold {color}]■  Segment ended[/bold {color}]  "
        f"reason=[bold]{reason}[/bold]  "
        f"dur=[bold yellow]{duration_s:.2f}s[/bold yellow]"
        + (f"  {extra}" if extra else "")
    )


def log_valley_found(time_s: float, score: float) -> None:
    console.print(
        f"[magenta]   🔍 Valley trough found[/magenta]  "
        f"at [bold]{time_s:.2f}s[/bold]  "
        f"score=[bold]{score:.3f}[/bold]"
    )


def log_no_valley(duration_s: float) -> None:
    if throttled("no_valley", 2.0):
        console.print(
            f"[dim yellow]   ⏳ No valley found yet at {duration_s:.1f}s — continuing…[/dim yellow]"
        )
