# vad_logging.py

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
from pathlib import Path

from jet.audio.speech.vad_types import ValleyTrough

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


def log_cond_1a(
    silence_frames: int, soft_limit_sec: float, frame_idx: int, duration: float
) -> None:
    if throttled("cond_1a", 2.0):
        console.print(
            f"[cyan][cond 1a][/cyan] Silence long enough ({silence_frames} frames), not past soft limit ({soft_limit_sec:.2f}s). [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_cond_1b(
    soft_limit_sec: float,
    in_silence: bool,
    silence_frames: int,
    frame_idx: int,
    duration: float,
) -> None:
    if throttled("cond_1b", 2.0):
        console.print(
            f"[cyan][cond 1b][/cyan] Silence while past soft limit (soft_limit_sec={soft_limit_sec:.2f}s), in_silence={in_silence}, silence_frames={silence_frames}. [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_cond_2a_check(
    soft_limit_sec: float, hard_limit_sec: float, frame_idx: int, duration: float
) -> None:
    if throttled("cond_2a_check", 2.0):
        console.print(
            f"[cyan][cond 2a][/cyan] Checking valley trough, past soft limit ({soft_limit_sec:.2f}s), not past hard limit ({hard_limit_sec:.2f}s). [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_cond_2a_valley_found(trough, frame_idx: int) -> None:
    if throttled("cond_2a_found", 2.0):
        console.print(
            f"[green][cond 2a match][/green] Valley trough found: {trough} [Frame: {frame_idx}]"
        )


def log_cond_2a_no_valley() -> None:
    if throttled("cond_2a_no_valley", 2.0):
        console.print("[dim][cond 2a] No valley trough found.[/dim]")


def log_cond_2b_check(hard_limit_sec: float, frame_idx: int, duration: float) -> None:
    if throttled("cond_2b_check", 2.0):
        console.print(
            f"[cyan][cond 2b][/cyan] Checking valley trough, past hard limit ({hard_limit_sec:.2f}s). [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_cond_2b_valley_found(trough, frame_idx: int) -> None:
    if throttled("cond_2b_found", 2.0):
        console.print(
            f"[green][cond 2b match][/green] Valley trough (relaxed) found: {trough} [Frame: {frame_idx}]"
        )


def log_cond_2b_no_valley_relaxed() -> None:
    if throttled("cond_2b_no_valley", 2.0):
        console.print(
            "[dim][cond 2b] No valley trough found (relaxed after hard limit).[/dim]"
        )


def log_cond_3(frame_idx: int, duration: float) -> None:
    if throttled("cond_3", 2.0):
        console.print(
            f"[red][cond 3][/red] Hard limit reached — emitting safety fallback. [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_cond_none(frame_idx: int, duration: float) -> None:
    if throttled("cond_none", 2.0):
        console.print(
            f"[dim][cond none][/dim] No end condition met. [Frame: {frame_idx}] [Duration: {duration:.2f}s]"
        )


def log_valley_found(valley_trough: ValleyTrough) -> None:
    score = valley_trough["valley"]["final_score"]
    console.print(
        f"[magenta]   🔍 Valley trough found[/magenta]  "
        f"at [bold]{valley_trough['time_s']:.2f}s[/bold]  "
        f"score=[bold]{score:.3f}[/bold]"
    )


def log_no_valley(duration_s: float) -> None:
    if throttled("no_valley", 2.0):
        console.print(
            f"[dim yellow]   ⏳ No valley found yet at {duration_s:.1f}s — continuing…[/dim yellow]"
        )


def linkify(path: str | Path):
    path = Path(path)
    # Provide clickable file link with basename (for rich/terminal that support it)
    return f"[bold blue][link=file://{path}]{path.name}[/link][/bold blue]"
