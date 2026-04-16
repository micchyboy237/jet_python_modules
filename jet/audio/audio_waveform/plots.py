import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


def create_plots_layout():
    """Create a widget containing a VAD selector dropdown and two plots.

    Returns:
        main_widget (QWidget): Container with dropdown and plots (top-level window).
        wave_curves (tuple): Three curves for audio amplitude.
        vad_curves (tuple): Three curves for selected VAD probability.
        vad_selector (QComboBox): Dropdown to choose VAD type.
        vad_plot (PlotItem): The VAD probability plot (for label updates).
    """
    # Main container widget (top-level window)
    main_widget = QtWidgets.QWidget()
    flags = QtCore.Qt.WindowType.Window | QtCore.Qt.WindowType.WindowStaysOnTopHint
    main_widget.setWindowFlags(flags)
    main_widget.setWindowTitle("Realtime Audio + VAD Probability")

    # Set size and position
    win_w, win_h = 450, 380
    margin = 20
    main_widget.resize(win_w, win_h)
    screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
    screen_w = screen.width()
    screen_h = screen.height()
    x = screen_w - win_w - margin
    y = screen_h - win_h - margin
    main_widget.move(x, y)

    # Layout for the main widget
    layout = QtWidgets.QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    main_widget.setLayout(layout)

    # Dropdown for VAD selection
    vad_selector = QtWidgets.QComboBox()
    vad_selector.addItems(["FireRed", "Silero", "SpeechBrain", "TEN VAD"])
    vad_selector.setCurrentIndex(0)  # FireRed default
    layout.addWidget(vad_selector)

    # GraphicsLayoutWidget for plots (child widget, no window flags)
    win = pg.GraphicsLayoutWidget()
    layout.addWidget(win)

    # Top plot: Audio Amp (waveform)
    wave_plot = win.addPlot()
    wave_plot.setYRange(0, 1.0)  # Normalized RMS is always [0, 1]
    wave_plot.setLabel("left", "Audio RMS")
    wave_plot.showGrid(x=True, y=True, alpha=0.15)
    wave_low = wave_plot.plot(pen=pg.mkPen(150, 150, 150, width=1.2), connect="finite")
    wave_mid = wave_plot.plot(pen=pg.mkPen(0, 255, 255, width=1.8), connect="finite")
    wave_high = wave_plot.plot(pen=pg.mkPen(100, 255, 120, width=2.2), connect="finite")

    # Next row: VAD Probability plot
    win.nextRow()
    vad_plot = win.addPlot()
    vad_plot.setYRange(0, 1)
    vad_plot.setLabel("left", "FireRed Prob")  # Default label
    vad_plot.showGrid(x=True, y=True, alpha=0.15)
    vad_low = vad_plot.plot(pen=pg.mkPen(255, 200, 120, width=1.2), connect="finite")
    vad_mid = vad_plot.plot(pen=pg.mkPen(255, 150, 80, width=1.8), connect="finite")
    vad_high = vad_plot.plot(pen=pg.mkPen(255, 100, 40, width=2.2), connect="finite")

    wave_curves = (wave_low, wave_mid, wave_high)
    vad_curves = (vad_low, vad_mid, vad_high)

    main_widget.show()
    return main_widget, wave_curves, vad_curves, vad_selector, vad_plot
