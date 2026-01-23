from __future__ import annotations

from pathlib import Path
import signal
import sys
import json
import datetime
from typing import Literal, List, Optional

from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView


LogLevel = Literal["debug", "info", "warning", "error", "success"]


class LoggerOverlay(QMainWindow):
    def __init__(self, html_path: str) -> None:
        super().__init__()

        self.setWindowTitle("Logger Overlay")
        self.resize(560, 420)

        self.view = QWebEngineView(self)
        self.setCentralWidget(self.view)

        self._pending_logs: List[str] = []
        self._js_ready = False

        self.view.loadFinished.connect(self._on_load_finished)
        self.view.load(QUrl.fromLocalFile(html_path))

    def _on_load_finished(self, ok: bool) -> None:
        if not ok:
            return

        # Check when JS declares itself ready
        self.view.page().runJavaScript(
            "window.__LOGGER_READY__ === true;",
            self._on_js_ready_check,
        )

    def _on_js_ready_check(self, ready: bool) -> None:
        if not ready:
            # Retry shortly (no busy loop)
            self.view.page().runJavaScript(
                "setTimeout(() => true, 50);",
                lambda _: self._on_load_finished(True),
            )
            return

        self._js_ready = True

        # Flush queued logs
        for js in self._pending_logs:
            self.view.page().runJavaScript(js)

        self._pending_logs.clear()

    def log(self, level: LogLevel, message: str) -> None:
        payload = json.dumps({
            "level": level,
            "message": message,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        })

        js = f"window.addLog({payload});"

        if self._js_ready:
            self.view.page().runJavaScript(js)
        else:
            self._pending_logs.append(js)

    @staticmethod
    def create_logger(html_path: Optional[str] = None) -> "Logger":
        """
        Factory helper:
        - creates QApplication if needed
        - installs Ctrl+C (SIGINT) handler
        - shows the overlay window
        - returns a Logger instance
        """

        if html_path is None:
            # Use default html
            html_path = str((Path(__file__).parent / "logger.html").resolve())

        # Ensure QApplication exists
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Ctrl+C handler
        def _handle_sigint(sig, frame):
            print("Exiting logger overlay (SIGINT)")
            QApplication.quit()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_sigint)

        # Create and show overlay
        overlay = LoggerOverlay(html_path)
        overlay.show()

        return Logger(overlay)

class Logger:
    def __init__(self, overlay: LoggerOverlay) -> None:
        self._overlay = overlay

    def debug(self, msg: str) -> None:
        self._overlay.log("debug", msg)

    def info(self, msg: str) -> None:
        self._overlay.log("info", msg)

    def warning(self, msg: str) -> None:
        self._overlay.log("warning", msg)

    def error(self, msg: str) -> None:
        self._overlay.log("error", msg)

    def success(self, msg: str) -> None:
        self._overlay.log("success", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    logger = LoggerOverlay.create_logger()

    logger.debug("Debug message")
    logger.info("Info message")
    logger.success("Success message")
    logger.warning("Warning message")
    logger.error("Error message")

    sys.exit(QApplication.exec())
