"""
Action handlers for SubtitleOverlay: copy, open folder, play audio.
Extracted as a mixin to keep SubtitleOverlay focused on window management.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtWidgets import QApplication


class SubtitleOverlayActions:
    """Mixin providing copy/open/play action handlers for SubtitleOverlay."""

    # Dependencies expected on the host instance:
    #   self._entries: list[dict]
    #   self._sound_effect: QSoundEffect
    #   self._WINDOW_TITLE: str
    #   setWindowTitle

    def _handle_anchor_click(self, url) -> None:
        url_str = url.toString()
        if url_str.startswith("copy:"):
            self._action_copy(int(url_str.split(":")[1]) - 1)
        elif url_str.startswith("open:"):
            self._action_open(int(url_str.split(":")[1]) - 1)
        elif url_str.startswith("play:"):
            self._action_play(int(url_str.split(":")[1]) - 1)

    def _action_copy(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        e = self._entries[idx]
        text = f"{e.get('ja', '')}\n{e.get('en', '')}".strip()
        QApplication.clipboard().setText(text)
        self.setWindowTitle("Copied ✓")
        QTimer.singleShot(800, lambda: self.setWindowTitle(self._WINDOW_TITLE))

    def _action_open(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        segment_dir = self._entries[idx].get("segment_dir")
        if segment_dir:
            try:
                subprocess.Popen(["open", str(segment_dir)])
            except Exception:
                pass

    def _action_play(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            return
        segment_dir = self._entries[idx].get("segment_dir")
        if not segment_dir:
            return
        wav_path = Path(segment_dir) / "sound.wav"
        if not wav_path.exists():
            return
        url = QUrl.fromLocalFile(str(wav_path))
        self._sound_effect.setSource(url)
        self._sound_effect.setVolume(1.0)
        self._sound_effect.play()
