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

        # Thresholds for coloring waveform and probability
        # Waveform ranges:
        #   < WAVE_MEDIUM  → gray
        #   WAVE_MEDIUM ≤ x < WAVE_HIGH  → cyan
        #   ≥ WAVE_HIGH              → green
        self.THRES_WAVE_MEDIUM = 0.01
        self.THRES_WAVE_HIGH = 0.15

        # Speech probability ranges
        self.THRES_PROB_MEDIUM = 0.01
        self.THRES_PROB_HIGH = 0.15

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
        self.app.setQuitOnLastWindowClosed(True)

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
        self.wave_plot = self.win.addPlot()
        self.wave_plot.setYRange(0, 1.1)
        self.wave_plot.setLabel("left", "Audio Amp")
        self.wave_plot.showGrid(x=True, y=True, alpha=0.15)  # Add faint grid
        self.wave_curve_low = self.wave_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.2),
            name="low",
            connect="finite",
        )
        self.wave_curve_mid = self.wave_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=1.8),
            name="mid",
            connect="finite",
        )
        self.wave_curve_high = self.wave_plot.plot(
            pen=pg.mkPen(color=(100, 255, 120), width=2.2),
            name="high",
            connect="finite",
        )

        # Move to next row
        self.win.nextRow()

        # -------------------------
        # Speech Probability plot (BOTTOM)
        # -------------------------
        self.prob_plot = self.win.addPlot()
        self.prob_plot.setYRange(0, 1)
        self.prob_plot.setLabel("left", "Speech Prob")
        self.prob_plot.showGrid(x=True, y=True, alpha=0.15)  # Add faint grid
        self.prob_curve_low = self.prob_plot.plot(
            pen=pg.mkPen(color=(150, 150, 150), width=1.2),
            name="low",
            connect="finite",
        )
        self.prob_curve_mid = self.prob_plot.plot(
            pen=pg.mkPen(color=(0, 255, 255), width=1.8),
            name="mid",
            connect="finite",
        )
        self.prob_curve_high = self.prob_plot.plot(
            pen=pg.mkPen(color=(100, 255, 120), width=2.2),
            name="high",
            connect="finite",
        )
        # Optional: keep for debugging
        # self.prob_plot.addLegend()

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
        wave_value = np.max(np.abs(samples)) if samples.size > 0 else 0.0
        self.wave_buffer.append(wave_value)

        # Speech probability (existing logic)
        energy = np.mean(np.abs(samples))
        prob = min(1.0, energy * 5.0)  # ← you can later replace with real VAD
        self.prob_buffer.append(prob)

    # -------------------------------------------------------------------------

    def _update_plots(self) -> None:
        wave_data = self.wave_buffer.to_array()
        if len(wave_data) > 0:
            x = np.arange(len(wave_data), dtype=np.float32)

            # Create masks
            low_mask = wave_data < self.THRES_WAVE_MEDIUM
            mid_mask = (wave_data >= self.THRES_WAVE_MEDIUM) & (
                wave_data < self.THRES_WAVE_HIGH
            )
            high_mask = wave_data >= self.THRES_WAVE_HIGH

            # Ensure continuity by overlapping boundary points
            for mask in (low_mask, mid_mask, high_mask):
                mask[:-1] |= mask[1:]
                mask[1:] |= mask[:-1]

            low = np.where(low_mask, wave_data, np.nan)
            mid = np.where(mid_mask, wave_data, np.nan)
            high = np.where(high_mask, wave_data, np.nan)

            self.wave_curve_low.setData(x, low)
            self.wave_curve_mid.setData(x, mid)
            self.wave_curve_high.setData(x, high)

        prob_data = self.prob_buffer.to_array()
        if len(prob_data) > 0:
            x_prob = np.arange(len(prob_data), dtype=np.float32)

            # Create masks
            low_mask = prob_data < self.THRES_PROB_MEDIUM
            mid_mask = (prob_data >= self.THRES_PROB_MEDIUM) & (
                prob_data < self.THRES_PROB_HIGH
            )
            high_mask = prob_data >= self.THRES_PROB_HIGH

            # Ensure continuity by overlapping boundary points
            for mask in (low_mask, mid_mask, high_mask):
                mask[:-1] |= mask[1:]
                mask[1:] |= mask[:-1]

            low = np.where(low_mask, prob_data, np.nan)
            mid = np.where(mid_mask, prob_data, np.nan)
            high = np.where(high_mask, prob_data, np.nan)

            self.prob_curve_low.setData(x_prob, low)
            self.prob_curve_mid.setData(x_prob, mid)
            self.prob_curve_high.setData(x_prob, high)

    # -------------------------------------------------------------------------

    def start(self) -> None:
        with self.stream:
            self.app.exec()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app = AudioWaveformWithSpeechProbApp()
    app.start()
