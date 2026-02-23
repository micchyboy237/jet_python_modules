"""
Realtime audio waveform + speech probability visualizer.

Two stacked plots:
1. Waveform
2. Speech probability (0–1)

Dependencies:
    pip install sounddevice numpy pyqtgraph
"""

from __future__ import annotations

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
        window_ms: int = 200,
    ) -> None:
        self.samplerate = samplerate
        self.block_size = block_size
        self.window_samples = int(window_ms * samplerate / 1000)

        # Buffers
        self.wave_buffer = CircularBuffer(self.window_samples)
        self.prob_buffer = CircularBuffer(200)  # fewer points needed

        # Qt App
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        # Enable OpenGL for smoother performance
        pg.setConfigOptions(useOpenGL=True)

        # Layout window
        self.win = pg.GraphicsLayoutWidget(
            show=True, title="Realtime Audio + Speech Probability"
        )

        # -------------------------
        # Waveform plot (TOP)
        # -------------------------
        self.wave_plot = self.win.addPlot(title="Waveform")
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Time", units="s")
        self.wave_curve = self.wave_plot.plot(pen="c")

        # Move to next row
        self.win.nextRow()

        # -------------------------
        # Speech Probability plot (BOTTOM)
        # -------------------------
        self.prob_plot = self.win.addPlot(title="Speech Probability")
        self.prob_plot.setYRange(0, 1)
        self.prob_plot.setLabel("left", "Probability")
        self.prob_plot.setLabel("bottom", "Frames")
        self.prob_curve = self.prob_plot.plot(pen="y")

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

        samples = indata[:, 0]
        self.wave_buffer.append(samples)

        # -----------------------------------------------------
        # Placeholder speech probability computation
        # Replace this with real VAD / model output
        # -----------------------------------------------------
        energy = np.mean(np.abs(samples))
        prob = min(1.0, energy * 5.0)  # simple normalized proxy

        self.prob_buffer.append(prob)

    # -------------------------------------------------------------------------

    def _update_plots(self) -> None:
        # Update waveform
        wave_data = self.wave_buffer.to_array()
        if len(wave_data) > 0:
            x = np.linspace(
                -len(wave_data) / self.samplerate,
                0,
                len(wave_data),
                dtype=np.float32,
            )
            self.wave_curve.setData(x, wave_data)

        # Update probability plot
        prob_data = self.prob_buffer.to_array()
        if len(prob_data) > 0:
            x_prob = np.arange(len(prob_data))
            self.prob_curve.setData(x_prob, prob_data)

    # -------------------------------------------------------------------------

    def start(self) -> None:
        with self.stream:
            self.app.exec()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app = AudioWaveformWithSpeechProbApp()
    app.start()
