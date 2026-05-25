"""
SegmentStore
============
Owns all on-disk state for one recording session:
    OUTPUT_DIR/
    └── segments/
        ├── segment_001/
        │   ├── sound.wav
        │   ├── speech_probs.json      ← from save_segments (replaces probs.json)
        │   ├── energies.json          ← from save_segments
        │   ├── hybrid_probs.json      ← from save_segments
        │   ├── speech_and_rms.png     ← from save_segments (replaces plot.png)
        │   ├── metadata.json          ← SegmentStore extra
        │   ├── summary.json           ← SegmentStore extra
        │   ├── vad_score.json         ← SegmentStore extra
        │   └── best_valley_trough.json ← SegmentStore extra (when present)
        └── segment_002/ ...

Public API
----------
    store = SegmentStore(segment_root)
    seg_dir, seg_num = store.save(speech_seg, audio_np, sample_rate)
    store.reset()          # wipe segment_root, restart numbering from 1
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.audio_waveform.vad.vad_utils import save_segment
from jet.audio.helpers.config import SAMPLE_RATE
from jet.audio.speech.segment_utils import build_summary
from jet.audio.speech.vad_types import ValleyTrough
from rich.console import Console
from rich.text import Text

console = Console()


def _save_best_valley_trough(
    seg_dir: Path,
    best_valley_trough: Optional[ValleyTrough],
) -> Optional[Path]:
    """
    Write ``seg_dir/best_valley_trough.json`` when a valley trough is present.

    Returns the written path, or None if the value is absent or serialisation
    fails.
    """
    if best_valley_trough is None:
        return None

    try:
        payload = dict(best_valley_trough)
    except Exception as exc:
        console.print(
            f"[yellow][SegmentStore] best_valley_trough serialisation failed: {exc}[/yellow]"
        )
        return None

    trough_path = seg_dir / "best_valley_trough.json"
    trough_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return trough_path


class SegmentStore:
    """
    Manages the segments/ directory for one recording session.

    Parameters
    ----------
    segment_root:
        Directory under which segment_NNN sub-dirs are created.
        Created automatically if it does not exist.
    """

    def __init__(self, segment_root: Path) -> None:
        self._root = segment_root
        self._next_num: int = self._scan_next_number()

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        speech_seg: SpeechSegment,
        seg_audio_np: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
    ) -> tuple[Path, int]:
        """
        Persist one speech segment to disk and return (seg_dir, seg_number).
        """
        self._root.mkdir(parents=True, exist_ok=True)

        seg_number = self._next_num
        self._next_num += 1
        speech_seg["num"] = seg_number

        # Build the segment_NNN directory first, then pass it directly to save_segment
        seg_dir = self._root / f"segment_{seg_number:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        saved = save_segment(
            speech_seg,
            audio_np=seg_audio_np,
            seg_dir=seg_dir,  # ← was incorrectly output_base_dir
            is_already_hybrid=True,
        )

        if not saved:
            raise RuntimeError(
                f"[SegmentStore] save_segments returned nothing for segment {seg_number}."
            )

        metadata_path = seg_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(dict(speech_seg), indent=2), encoding="utf-8"
        )

        summary = build_summary(speech_seg, seg_audio_np, seg_number)
        summary_path = seg_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        best_valley_trough_path = _save_best_valley_trough(
            seg_dir,
            speech_seg.get("best_valley_trough"),
        )

        console.print(
            f"\n[green]Segment {seg_number} saved to:[/green] ",
            Text(
                seg_dir.name,
                style=f"bold bright_green link file://{seg_dir}",
            ),
        )

        logged_paths: list[Path] = [
            seg_dir / "sound.wav",
            metadata_path,
            summary_path,
            seg_dir / "speech_and_rms.png",
        ]

        if best_valley_trough_path is not None:
            logged_paths.append(best_valley_trough_path)

        for p in logged_paths:
            console.print(Text(p.name, style=f"bold bright_green link file://{p}"))

        return seg_dir, seg_number

    def reset(self) -> None:
        """
        Delete all segments and restart numbering from 1.

        Safe to call while a recording is in progress — the next call to
        save() will simply create segment_001 again.
        """
        try:
            shutil.rmtree(self._root, ignore_errors=True)
            self._root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            console.print(f"[red][SegmentStore] Failed to reset: {exc}[/red]")
        self._next_num = 1
        console.print("[green][SegmentStore] Reset — next segment will be 001[/green]")

    def _scan_next_number(self) -> int:
        """
        Scan existing segment_NNN dirs and return the next free number.

        Called once at construction so subsequent saves are O(1) in-memory
        increments rather than repeated directory scans.
        """
        self._root.mkdir(parents=True, exist_ok=True)
        used = {
            int(d.name.split("_")[1])
            for d in self._root.glob("segment_*")
            if d.name.split("_")[1].isdigit()
        }
        num = 1
        while num in used:
            num += 1
        return num
