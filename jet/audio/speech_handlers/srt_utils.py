"""
Utilities for building and writing SRT subtitle files.
Pure functions — no I/O side-effects except write_srt().
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def seconds_to_srt_time(total_seconds: float) -> str:
    """Convert a float seconds value to SRT timestamp: HH:MM:SS,mmm."""
    total_seconds = max(0.0, total_seconds)
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = int(total_seconds % 60)
    millis = int(round((total_seconds - int(total_seconds)) * 1000))
    # clamp millis overflow
    if millis >= 1000:
        millis = 999
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_block(
    index: int,
    start_sec: float,
    end_sec: float,
    lines: Sequence[str],
) -> str:
    """Return one SRT block as a string (no trailing newline)."""
    text = "\n".join(line for line in lines if line.strip())
    return (
        f"{index}\n"
        f"{seconds_to_srt_time(start_sec)} --> {seconds_to_srt_time(end_sec)}\n"
        f"{text}"
    )


def build_srt_from_phrase_segments(
    phrase_segments: list[dict],
    global_start_sec: float = 0.0,
) -> str:
    """
    Build a complete SRT string from the server's phrase_segments list.
    Each item: {"phrase": str, "start": float, "end": float}
    start/end are relative to the segment; global_start_sec shifts them.
    """
    blocks: list[str] = []
    for i, ps in enumerate(phrase_segments, start=1):
        phrase = ps.get("phrase", "").strip()
        if not phrase:
            continue
        abs_start = global_start_sec + float(ps.get("start", 0.0))
        abs_end = global_start_sec + float(ps.get("end", 0.0))
        blocks.append(build_srt_block(i, abs_start, abs_end, [phrase]))
    return "\n\n".join(blocks)


def build_srt_single_block(
    index: int,
    start_sec: float,
    end_sec: float,
    transcription_ja: str,
    translation_en: str,
) -> str:
    """
    Fallback: one SRT block covering the whole segment.
    Japanese on line 1, English on line 2.
    """
    lines = [l for l in [transcription_ja.strip(), translation_en.strip()] if l]
    return build_srt_block(index, start_sec, end_sec, lines)


def write_srt(path: Path, content: str) -> None:
    """Write SRT content to *path*, ensuring UTF-8 BOM for maximum player compat."""
    path.write_text(content + "\n", encoding="utf-8-sig")
