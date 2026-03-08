import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


def create_plots_layout():
    """Create the four-plot layout and return window + curve tuples"""

    flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint

    win = pg.GraphicsLayoutWidget(
        size=(450, 400),
        title="Realtime Audio + Speech Probability",
    )
    win.setWindowFlags(flags)

    # ── Waveform ────────────────────────────────────────────────
    wave_plot = win.addPlot()
    wave_plot.setYRange(0, 1.1)
    wave_plot.setLabel("left", "Audio Amp")
    wave_plot.showGrid(x=True, y=True, alpha=0.15)

    wave_low = wave_plot.plot(pen=pg.mkPen(150, 150, 150, width=1.2), connect="finite")
    wave_mid = wave_plot.plot(pen=pg.mkPen(0, 255, 255, width=1.8), connect="finite")
    wave_high = wave_plot.plot(pen=pg.mkPen(100, 255, 120, width=2.2), connect="finite")

    win.nextRow()

    # ── Silero Prob ─────────────────────────────────────────────
    prob_plot = win.addPlot()
    prob_plot.setYRange(0, 1)
    prob_plot.setLabel("left", "Speech Prob")
    prob_plot.showGrid(x=True, y=True, alpha=0.15)

    p_low = prob_plot.plot(pen=pg.mkPen(150, 150, 150, width=1.2), connect="finite")
    p_mid = prob_plot.plot(pen=pg.mkPen(0, 255, 255, width=1.8), connect="finite")
    p_high = prob_plot.plot(pen=pg.mkPen(100, 255, 120, width=2.2), connect="finite")

    win.nextRow()

    # ── SpeechBrain Prob ────────────────────────────────────────
    sb_plot = win.addPlot()
    sb_plot.setYRange(0, 1)
    sb_plot.setLabel("left", "SB Speech Prob")
    sb_plot.showGrid(x=True, y=True, alpha=0.15)

    sb_low = sb_plot.plot(pen=pg.mkPen(180, 150, 180, width=1.2))
    sb_mid = sb_plot.plot(pen=pg.mkPen(200, 100, 200, width=1.8))
    sb_high = sb_plot.plot(pen=pg.mkPen(220, 60, 220, width=2.2))

    win.nextRow()

    # ── FireRed Prob ────────────────────────────────────────────
    fr_plot = win.addPlot()
    fr_plot.setYRange(0, 1)
    fr_plot.setLabel("left", "FR Speech Prob")
    fr_plot.showGrid(x=True, y=True, alpha=0.15)

    fr_low = fr_plot.plot(pen=pg.mkPen(255, 200, 120, width=1.2))
    fr_mid = fr_plot.plot(pen=pg.mkPen(255, 150, 80, width=1.8))
    fr_high = fr_plot.plot(pen=pg.mkPen(255, 100, 40, width=2.2))

    wave_curves = (wave_low, wave_mid, wave_high)
    prob_curves = (p_low, p_mid, p_high)
    sb_curves = (sb_low, sb_mid, sb_high)
    fr_curves = (fr_low, fr_mid, fr_high)

    return win, (wave_curves, prob_curves, sb_curves, fr_curves)
