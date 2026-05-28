"""
Utilities for building and writing SRT subtitle files.
Pure functions — no I/O side-effects except write_srt() and merge_and_write_global_srt().
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


def build_segment_srt(
    index: int,
    start_sec: float,
    end_sec: float,
    ja_text: str,
    en_text: str,
) -> str:
    """
    One SRT block for a whole segment.
    Japanese on line 1, English on line 2.
    index is the SRT sequence number (1-based, globally unique).
    """
    lines = [ln for ln in [ja_text.strip(), en_text.strip()] if ln]
    return build_srt_block(index, start_sec, end_sec, lines)


def write_srt(path: Path, content: str) -> None:
    """Write SRT content to *path* with UTF-8 BOM for maximum player compat."""
    path.write_text(content + "\n", encoding="utf-8-sig")


def merge_and_write_global_srt(
    segments_root: Path,
    global_srt_path: Path,
) -> None:
    """
    Scan every segment_NNN/ subdir under segments_root for a subtitle.srt,
    collect them in segment number order, re-index blocks sequentially (1, 2, 3 …),
    and write the merged result to global_srt_path.

    Called after each segment finishes so the global file is always up-to-date.
    Skips segments whose subtitle.srt does not exist yet (still in-flight).
    """
    seg_dirs = sorted(
        (
            d
            for d in segments_root.iterdir()
            if d.is_dir() and d.name.startswith("segment_")
        ),
        key=lambda d: int(d.name.split("_")[1]),
    )

    blocks: list[str] = []
    global_index = 1

    for seg_dir in seg_dirs:
        srt_file = seg_dir / "subtitle.srt"
        if not srt_file.exists():
            continue
        raw = srt_file.read_text(encoding="utf-8-sig").strip()
        if not raw:
            continue
        # Each segment srt is a single block: line 0 = local index,
        # line 1 = timestamps, lines 2+ = text. Re-number the index.
        lines = raw.splitlines()
        if len(lines) < 3:
            continue
        reindexed = "\n".join([str(global_index)] + lines[1:])
        blocks.append(reindexed)
        global_index += 1

    write_srt(global_srt_path, "\n\n".join(blocks))
