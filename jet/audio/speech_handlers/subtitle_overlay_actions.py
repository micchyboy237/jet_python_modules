"""
Action handlers for SubtitleOverlay: copy, open folder, play audio.
Extracted as a mixin to keep SubtitleOverlay focused on window management.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from jet.logger import logger
from PyQt6.QtCore import QTimer, QUrl
from PyQt6.QtWidgets import QApplication


class SubtitleOverlayActions:
    """Mixin providing copy/open/play action handlers for SubtitleOverlay."""

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
            logger.warning(
                f"[SubtitleOverlay] Copy action: invalid index {idx}, "
                f"entries count: {len(self._entries)}"
            )
            return
        e = self._entries[idx]
        text = f"{e.get('ja', '')}\n{e.get('en', '')}".strip()
        if not text:
            logger.warning(
                f"[SubtitleOverlay] Copy action: empty text for segment "
                f"#{e.get('segment_number', 'unknown')}"
            )
        QApplication.clipboard().setText(text)
        self.setWindowTitle("Copied ✓")
        QTimer.singleShot(800, lambda: self.setWindowTitle(self._WINDOW_TITLE))
        logger.debug(
            f"[SubtitleOverlay] Copied text for segment "
            f"#{e.get('segment_number', 'unknown')}"
        )

    def _action_open(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            logger.warning(
                f"[SubtitleOverlay] Open action: invalid index {idx}, "
                f"entries count: {len(self._entries)}"
            )
            return
        segment_dir = self._entries[idx].get("segment_dir")
        if not segment_dir:
            seg_num = self._entries[idx].get("segment_number", "unknown")
            logger.warning(
                f"[SubtitleOverlay] Open action: no segment_dir for "
                f"segment #{seg_num} at index {idx}"
            )
            return
        dir_path = Path(segment_dir)
        if not dir_path.exists():
            logger.error(
                f"[SubtitleOverlay] Open action: directory does not exist: {dir_path}"
            )
            return
        try:
            subprocess.Popen(["open", str(dir_path)])
            logger.info(
                f"[SubtitleOverlay] Opened directory for segment "
                f"#{self._entries[idx].get('segment_number', 'unknown')}: {dir_path}"
            )
        except Exception as exc:
            logger.error(
                f"[SubtitleOverlay] Failed to open directory {dir_path}: {exc}"
            )

    def _action_play(self, idx: int) -> None:
        if not (0 <= idx < len(self._entries)):
            logger.warning(
                f"[SubtitleOverlay] Play action: invalid index {idx}, "
                f"entries count: {len(self._entries)}"
            )
            return
        segment_dir = self._entries[idx].get("segment_dir")
        seg_num = self._entries[idx].get("segment_number", "unknown")
        if not segment_dir:
            logger.warning(
                f"[SubtitleOverlay] Play action: no segment_dir for "
                f"segment #{seg_num} at index {idx}"
            )
            return
        wav_path = Path(segment_dir) / "sound.wav"
        if not wav_path.exists():
            logger.error(
                f"[SubtitleOverlay] Play action: WAV file not found: {wav_path} "
                f"(segment #{seg_num})"
            )
            return
        url = QUrl.fromLocalFile(str(wav_path))
        self._sound_effect.setSource(url)
        self._sound_effect.setVolume(1.0)
        self._sound_effect.play()
        logger.info(
            f"[SubtitleOverlay] Playing audio for segment #{seg_num}: {wav_path}"
        )
