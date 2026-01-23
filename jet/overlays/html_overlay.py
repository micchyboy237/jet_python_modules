# local_html_viewer.py
from __future__ import annotations

import sys
import os
from typing import Optional

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl


class LocalHtmlViewer(QMainWindow):
    """
    A simple QMainWindow that displays local HTML files using QWebEngineView.
    
    Supports full modern HTML/CSS/JS (via Chromium engine).
    Loads via file:// URL for perfect relative resource resolution.
    """
    
    def __init__(
        self,
        file_path: str = "index.html",
        title: str = "Local HTML Viewer",
        initial_size: tuple[int, int] = (1100, 850),
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(*initial_size)

        self.browser = QWebEngineView()
        self.setCentralWidget(self.browser)

        self._load_html(file_path)

    def _load_html(self, file_path: str) -> None:
        """Load the HTML file from the given path (or show error)."""
        # Convert to absolute path — safest for QUrl.fromLocalFile
        abs_path = os.path.abspath(file_path)

        if not os.path.isfile(abs_path):
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <body style="font-family: system-ui; padding: 3rem; background: #fef2f2; color: #991b1b;">
                <h1>File Not Found</h1>
                <p>Could not find: <code>{abs_path}</code></p>
                <p>Make sure the file exists and the path is correct.</p>
            </body>
            </html>
            """
            self.browser.setHtml(error_html)
            return

        url = QUrl.fromLocalFile(abs_path)
        self.browser.load(url)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Allow passing path as command-line argument (optional)
    file_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/overlays/sample.html"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]

    window = LocalHtmlViewer(
        file_path=file_path,
        title=f"Local Viewer — {os.path.basename(file_path)}",
    )
    window.show()
    sys.exit(app.exec())