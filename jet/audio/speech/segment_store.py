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
        │   ├── plot.png
        │   └── vad_score.json     ← NEW: VADScorer metrics
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
from jet.audio.audio_waveform.vad.vad_scorer import VADScorer
from jet.audio.helpers.config import FRAME_SHIFT_S, SAMPLE_RATE
from jet.audio.speech.segment_utils import build_summary, save_segment_plot
from jet.audio.speech.wav_utils import save_wav_file
from rich.console import Console
from rich.text import Text

console = Console()


def _save_vad_score(
    seg_dir: Path,
    segment_probs: list[float],
) -> Optional[Path]:
    """
    Run VADScorer on *segment_probs* and write the result to
    ``seg_dir/vad_score.json``.

    Returns the written path, or None if probs are empty / scoring fails.

    Parameters
    ----------
    seg_dir:
        Directory for the current segment (must already exist).
    segment_probs:
        Per-frame VAD probability list, values in [0, 1].
    """
    if not segment_probs:
        return None

    try:
        scorer = VADScorer(
            probs=segment_probs,
            frame_shift_s=FRAME_SHIFT_S,
        )
        metrics = scorer.summary()  # plain serialisable dict
    except Exception as exc:
        console.print(f"[yellow][SegmentStore] VADScorer failed: {exc}[/yellow]")
        return None

    vad_score_path = seg_dir / "vad_score.json"
    vad_score_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return vad_score_path


class SegmentStore:
    """
    Manages the segments/ directory for one recording session.

    Parameters
    ----------
    segment_root:
        Directory under which segment_NNN sub-dirs are created.
        Created automatically if it does not exist.
    """

    def __init__(
        self,
        segment_root: Path,
    ) -> None:
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

        Written files
        -------------
        sound.wav        — raw PCM audio
        metadata.json    — raw SpeechSegment dict
        summary.json     — human-readable audio / VAD insights
        plot.png         — 3-panel diagnostic plot
        vad_score.json   — VADScorer metrics (composite_score, quality_label, …)
                           written only when segment_probs is non-empty
        """
        self._root.mkdir(parents=True, exist_ok=True)

        seg_number = self._next_num
        self._next_num += 1

        seg_dir = self._root / f"segment_{seg_number:03d}"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # ── audio ────────────────────────────────────────────────────────────
        wav_path = seg_dir / "sound.wav"
        seg_sound_file = save_wav_file(wav_path, seg_audio_np)

        # ── metadata (raw SpeechSegment) ─────────────────────────────────────
        metadata_path = seg_dir / "metadata.json"
        metadata_path.write_text(json.dumps(speech_seg, indent=2), encoding="utf-8")

        # ── summary (human-readable insights) ────────────────────────────────
        summary = build_summary(speech_seg, seg_audio_np, seg_number)
        summary_path = seg_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # ── diagnostic plot ───────────────────────────────────────────────────
        plot_path = seg_dir / "plot.png"
        save_segment_plot(speech_seg, seg_audio_np, seg_number, plot_path, sample_rate)

        # ── VAD score ─────────────────────────────────────────────────────────
        segment_probs: list[float] = speech_seg.get("segment_probs") or []
        vad_score_path = _save_vad_score(
            seg_dir,
            segment_probs,
        )

        # ── console log ───────────────────────────────────────────────────────
        console.print(
            f"\n[green]Segment {seg_number} saved to:[/green] ",
            Text(
                Path(seg_dir).name,
                style=f"bold bright_green link file://{seg_dir}",
            ),
        )
        logged_paths = [seg_sound_file, metadata_path, summary_path, plot_path]
        if vad_score_path is not None:
            logged_paths.append(vad_score_path)

        for p in logged_paths:
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
