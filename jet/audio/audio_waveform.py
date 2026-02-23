"""
Realtime audio waveform + speech probability visualizer.

Two stacked plots:
1. Waveform
2. Speech probability (0–1)

Dependencies:
    pip install sounddevice numpy pyqtgraph
"""

from __future__ import annotations

import signal
import sys
from collections import deque
from typing import Deque

import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from pyqtgraph.Qt import QtCore, QtWidgets

# -----------------------------------------------------------------------------
# Circular Buffer Base
# -----------------------------------------------------------------------------


class CircularBuffer:
    """Generic fixed-length circular buffer."""

    def __init__(self, max_len: int) -> None:
        if max_len <= 0:
            raise ValueError("max_len must be > 0")

        self.max_len = max_len
        self._buffer: Deque[float] = deque(maxlen=max_len)

    def append(self, values: np.ndarray | float) -> None:
        if isinstance(values, np.ndarray):
            for v in values:
                self._buffer.append(float(v))
        else:
            self._buffer.append(float(values))

    def to_array(self) -> np.ndarray:
        return np.array(self._buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)


# -----------------------------------------------------------------------------
# Main App
# -----------------------------------------------------------------------------


class AudioWaveformWithSpeechProbApp:
    """
    Displays:
        - Waveform (top)
        - Speech probability (bottom)
    """

    def __init__(
        self,
        samplerate: int = 16000,
        block_size: int = 512,
        display_points: int = 200,  # ≈ 6.4 seconds @ 512 samples / 32 ms per block
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.display_points = display_points

        # Buffers
        self.wave_buffer = CircularBuffer(display_points)
        self.prob_buffer = CircularBuffer(display_points)

        # Initialize with zeros for smooth startup visual
        for _ in range(display_points):
            self.wave_buffer.append(0.0)
            self.prob_buffer.append(0.0)

        # Qt App
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        def _quit_on_sigint(sig, frame):
            self.app.quit()

        signal.signal(signal.SIGINT, _quit_on_sigint)

        # Enable OpenGL for smoother performance
        pg.setConfigOptions(useOpenGL=True)

        # Prepare flags first
        flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint

        # Layout window
        self.win = pg.GraphicsLayoutWidget(
            show=False,  # ← important: do NOT show yet
            size=(450, 300),
            title="Realtime Audio + Speech Probability",
        )
        self.win.setWindowFlags(flags)

        # -------------------------
        # Waveform plot (TOP)
        # -------------------------
        self.wave_plot = self.win.addPlot(title="Waveform")
        self.wave_plot.setYRange(0, 1.1)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Recent blocks (~9.6 s)")
        self.wave_curve_zero = self.wave_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=2)
        )
        self.wave_curve_active = self.wave_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=2)
        )

        # Move to next row
        self.win.nextRow()

        # -------------------------
        # Speech Probability plot (BOTTOM)
        # -------------------------
        self.prob_plot = self.win.addPlot(title="Speech Probability")
        self.prob_plot.setYRange(0, 1)
        self.prob_plot.setLabel("left", "Probability")
        self.prob_plot.setLabel("bottom", "Recent blocks (~9.6 s)")
        self.prob_curve_zero = self.prob_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.5)
        )
        self.prob_curve_active = self.prob_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=1.5)
        )

        # Position at bottom-right corner with small margin
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        margin = 0
        x = screen.width() - self.win.width() - margin
        y = screen.height() - self.win.height() - 70 - margin
        self.win.move(max(0, x), max(0, y))  # prevent going off-screen left/top

        self.win.show()  # show explicitly after flags & position

        # Audio stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # UI timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(30)

    # -------------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)
        samples = indata[:, 0].astype(np.float32)

        # Waveform: one value per block → peak absolute amplitude
        wave_value = np.max(np.abs(samples)) if len(samples) > 0 else 0.0
        self.wave_buffer.append(wave_value)

        # Speech probability (existing logic)
        energy = np.mean(np.abs(samples))
        prob = min(1.0, energy * 5.0)
        self.prob_buffer.append(prob)

    # -------------------------------------------------------------------------

    def _update_plots(self) -> None:
        wave_data = self.wave_buffer.to_array()
        if len(wave_data) > 0:
            x = np.arange(len(wave_data), dtype=np.float32)

            # Split zero vs non-zero
            wave_zero = np.where(wave_data == 0.0, wave_data, np.nan)
            wave_active = np.where(wave_data != 0.0, wave_data, np.nan)

            self.wave_curve_zero.setData(x, wave_zero)
            self.wave_curve_active.setData(x, wave_active)

        prob_data = self.prob_buffer.to_array()
        if len(prob_data) > 0:
            x_prob = np.arange(len(prob_data), dtype=np.float32)

            prob_zero = np.where(prob_data == 0.0, prob_data, np.nan)
            prob_active = np.where(prob_data != 0.0, prob_data, np.nan)

            self.prob_curve_zero.setData(x_prob, prob_zero)
            self.prob_curve_active.setData(x_prob, prob_active)

    # -------------------------------------------------------------------------

    def start(self) -> None:
        with self.stream:
            self.app.exec()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app = AudioWaveformWithSpeechProbApp()
    app.start()
