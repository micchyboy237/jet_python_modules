import sys

import numpy as np
from jet.audio.audio_waveform.circular_buffer import CircularBuffer
from jet.audio.audio_waveform.plots import create_plots_layout
from pyqtgraph.Qt import QtCore, QtWidgets


class VisualizerObserver:
    def __init__(self, display_points: int = 200):
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        self.display_points = display_points
        self.THRES_WAVE = (0.01, 0.15)
        self.THRES_PROB = (0.3, 0.7)

        self.buffers = {
            "wave": CircularBuffer(display_points),
            "silero": CircularBuffer(display_points),
            "sb": CircularBuffer(display_points),
            "fr": CircularBuffer(display_points),
            "ten_vad": CircularBuffer(display_points),  # NEW: TEN-VAD
        }
        self._fill_initial_buffers()

        (
            self.win,
            (
                self.c_wave,
                self.c_silero,
                self.c_sb,
                self.c_fr,
                self.c_ten,  # NEW
            ),
        ) = create_plots_layout()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(30)

    def _fill_initial_buffers(self):
        for buf in self.buffers.values():
            for _ in range(self.display_points):
                buf.append(0.0)

    def __call__(self, samples: np.ndarray):
        pass

    def push_data(
        self, wave: float, silero: float, sb: float, fr: float, ten_vad: float
    ):
        """Updated to accept TEN-VAD probability"""
        self.buffers["wave"].append(wave)
        self.buffers["silero"].append(silero)
        self.buffers["sb"].append(sb)
        self.buffers["fr"].append(fr)
        self.buffers["ten_vad"].append(ten_vad)  # NEW

    def update_plots(self):
        self._update_curve(self.buffers["wave"], self.c_wave, *self.THRES_WAVE)
        self._update_curve(self.buffers["silero"], self.c_silero, *self.THRES_PROB)
        self._update_curve(self.buffers["sb"], self.c_sb, *self.THRES_PROB)
        self._update_curve(self.buffers["fr"], self.c_fr, *self.THRES_PROB)
        self._update_curve(self.buffers["ten_vad"], self.c_ten, *self.THRES_PROB)  # NEW

    def _update_curve(self, buffer, curves, t_med, t_high):
        data = buffer.to_array()
        if len(data) == 0:
            return
        x = np.arange(len(data), dtype=np.float32)
        low_m = data < t_med
        mid_m = (data >= t_med) & (data < t_high)
        high_m = data >= t_high

        for m in (low_m, mid_m, high_m):
            m[:-1] |= m[1:]
            m[1:] |= m[:-1]

        low_c, mid_c, high_c = curves
        low_c.setData(x, np.where(low_m, data, np.nan))
        mid_c.setData(x, np.where(mid_m, data, np.nan))
        high_c.setData(x, np.where(high_m, data, np.nan))

    def close(self):
        self.timer.stop()
        self.win.close()
        self.app.quit()
