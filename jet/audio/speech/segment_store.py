# jet.audio.speech.segment_store

"""
SegmentStore
============
Owns all on-disk state for one recording session:

    OUTPUT_DIR/
    └── segments/
        ├── segment_001/
        │   ├── sound.wav
        │   ├── metadata.json
        │   ├── summary.json
        │   └── plot.png
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

import numpy as np
from jet.audio.audio_waveform.vad._types import SpeechSegment
from jet.audio.speech.segment_utils import build_summary, save_segment_plot
from jet.audio.speech.wav_utils import save_wav_file
from rich.console import Console
from rich.text import Text

console = Console()


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def save(
        self,
        speech_seg: SpeechSegment,
        seg_audio_np: np.ndarray,
        sample_rate: int = 16_000,
    ) -> tuple[Path, int]:
        """
        Persist one speech segment to disk and return (seg_dir, seg_number).

        Written files
        -------------
        sound.wav       — raw PCM audio
        metadata.json   — raw SpeechSegment dict
        summary.json    — human-readable insights
        plot.png        — 3-panel diagnostic plot
        """
        self._root.mkdir(parents=True, exist_ok=True)

        seg_number = self._next_num
        self._next_num += 1

        seg_dir = self._root / f"segment_{seg_number:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        wav_path = seg_dir / "sound.wav"
        seg_sound_file = save_wav_file(wav_path, seg_audio_np)

        metadata_path = seg_dir / "metadata.json"
        metadata_path.write_text(json.dumps(speech_seg, indent=2), encoding="utf-8")

        summary = build_summary(speech_seg, seg_audio_np, seg_number)
        summary_path = seg_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        plot_path = seg_dir / "plot.png"
        save_segment_plot(speech_seg, seg_audio_np, seg_number, plot_path, sample_rate)

        console.print(
            f"\n[green]Segment {seg_number} saved to:[/green] ",
            Text(Path(seg_dir).name, style=f"bold bright_green link file://{seg_dir}"),
        )
        for p in (seg_sound_file, metadata_path, summary_path, plot_path):
            console.print(
                Text(Path(p).name, style=f"bold bright_green link file://{p}")
            )

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
