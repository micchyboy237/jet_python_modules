"""
Realtime audio waveform visualizer.

Modern, minimal, high-performance implementation using:
- sounddevice (audio capture)
- pyqtgraph (fast OpenGL plotting)
- QtWidgets QApplication (correct modern import)

Run:
    python audio_waveform.py
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
# AudioStreamingBuffer
# -----------------------------------------------------------------------------


class AudioStreamingBuffer:
    """
    Fixed-size circular buffer for streaming audio samples.
    """

    def __init__(self, buffer_len: int) -> None:
        if buffer_len <= 0:
            raise ValueError("buffer_len must be > 0")

        self.buffer_len: int = buffer_len
        self._buffer: Deque[float] = deque(maxlen=buffer_len)

    def append(self, samples: np.ndarray) -> None:
        """Append new audio samples."""
        if samples.ndim != 1:
            raise ValueError("Samples must be 1D array")

        for s in samples:
            self._buffer.append(float(s))

    def to_array(self) -> np.ndarray:
        """Return current buffer contents as numpy array."""
        return np.array(self._buffer, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._buffer)


# -----------------------------------------------------------------------------
# AudioWaveformApp
# -----------------------------------------------------------------------------


class AudioWaveformApp:
    """
    Captures microphone input and renders a scrolling waveform.
    """

    def __init__(
        self,
        samplerate: int = 44100,
        block_size: int = 1024,
        window_ms: int = 200,
    ) -> None:
        if samplerate <= 0:
            raise ValueError("samplerate must be > 0")

        self.samplerate = samplerate
        self.block_size = block_size
        self.window_samples = int(window_ms * samplerate / 1000)

        self.buffer = AudioStreamingBuffer(self.window_samples)

        # Create Qt application properly (FIX #1)
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)

        # Create main window
        self.win = pg.GraphicsLayoutWidget(show=True, title="Realtime Audio Waveform")

        # FIX #2: addPlot is valid here
        self.plot = self.win.addPlot(title="Waveform")
        self.plot.setYRange(-1, 1)
        self.plot.setLabel("left", "Amplitude")
        self.plot.setLabel("bottom", "Time", units="s")

        self.curve = self.plot.plot(pen="c")

        # Audio input stream
        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        # Timer for UI refresh (~33 FPS)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._update_plot)
        self.timer.start(30)

    # -------------------------------------------------------------------------

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print(status)

        samples = indata[:, 0]
        self.buffer.append(samples)

    # -------------------------------------------------------------------------

    def _update_plot(self) -> None:
        data = self.buffer.to_array()
        if len(data) == 0:
            return

        x = np.linspace(
            -len(data) / self.samplerate,
            0,
            len(data),
            dtype=np.float32,
        )

        self.curve.setData(x, data)

    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start streaming and Qt event loop."""
        with self.stream:
            self.app.exec()


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app = AudioWaveformApp()
    app.start()
