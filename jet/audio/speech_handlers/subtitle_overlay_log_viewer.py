"""
Per-segment status log viewer dialog for SubtitleOverlay.
Extracted to keep the window class under the size threshold.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)
from rich.console import Console

console = Console()


class SubtitleOverlayLogViewer:
    """Mixin providing the segment status log viewer dialog."""

    # Dependencies expected on the host instance:
    #   self._segment_statuses: dict[int, dict]
    #   self._segment_order: list[int]
    #   self._max_log_segments: int
    #   self._log_viewer: Optional[QDialog]
    #   self._queue_pending_count: int
    #   self.update_queue_status(status, pending, status_color)

    @staticmethod
    def _get_status_icon(
        status_text: str, final_status: Optional[str], info: dict = None
    ) -> str:
        """Determine the appropriate icon based on segment status."""
        if info and info.get("status") == "success":
            return "✅"
        if info and info.get("status") == "error":
            return "❌"
        if final_status == "success":
            return "✅"
        if final_status == "error":
            return "❌"
        if "⏳" in status_text or "sending" in status_text.lower():
            return "⏳"
        elif "🔄" in status_text or "retry" in status_text.lower():
            return "🔄"
        elif "✅" in status_text or "succeeded" in status_text.lower():
            return "✅"
        elif "❌" in status_text or "failed" in status_text.lower():
            return "❌"
        elif "📋" in status_text or "queued" in status_text.lower():
            return "📋"
        else:
            return "📡"

    @staticmethod
    def _get_status_color(final_status: Optional[str], current_color: str) -> str:
        """Get color based on final status."""
        if final_status == "success":
            return "#3fb950"
        elif final_status == "error":
            return "#f85149"
        return current_color

    def _generate_plain_text_log(self) -> str:
        """Generate a plain text version of the segment history for clipboard."""
        lines = []
        lines.append("=" * 70)
        lines.append("QUEUE STATUS HISTORY - SEGMENT LOG")
        lines.append(
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        lines.append(f"Total Segments: {len(self._segment_order)}")
        lines.append("=" * 70)
        lines.append("")
        for seg_num in self._segment_order:
            if seg_num not in self._segment_statuses:
                continue
            seg_info = self._segment_statuses[seg_num]
            duration = seg_info.get("duration")
            start_sec = seg_info.get("start_sec")
            retry_count = seg_info.get("retry_count", 0)
            final_status = seg_info.get("final_status", "pending")
            first_seen = seg_info["first_seen"]
            last_updated = seg_info["last_updated"]
            statuses = seg_info["statuses"]
            status_str = final_status.upper() if final_status else "PENDING"
            lines.append(f"[{status_str}] Segment #{seg_num}")
            if duration:
                lines.append(f"  Duration: {duration:.2f}s")
            if start_sec is not None:
                lines.append(f"  Start: {start_sec:.2f}s")
            if retry_count > 0:
                lines.append(f"  Retries: {retry_count}")
            lines.append(f"  Time: {first_seen} → {last_updated}")
            lines.append("")
            lines.append("  Status Timeline:")
            for entry in statuses:
                icon = self._get_status_icon(entry["status"], None, entry.get("info"))
                lines.append(f"    {entry['timestamp']} {icon} {entry['status']}")
                info = entry.get("info", {})
                if info.get("error"):
                    lines.append(f"      Error: {info['error']}")
                if info.get("retry_attempt"):
                    lines.append(
                        f"      Attempt: {info['retry_attempt']}, "
                        f"Delay: {info.get('retry_delay', 0):.1f}s"
                    )
            lines.append("")
            lines.append("-" * 70)
            lines.append("")
        return "\n".join(lines)

    def _show_log_history(self) -> None:
        """Show a dialog with per-segment status history that updates live."""
        dialog = QDialog(self)
        dialog.setWindowTitle("📜 Queue Status History ● LIVE")
        dialog.resize(650, 550)
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        live_indicator = QLabel("● LIVE")
        live_indicator.setStyleSheet(
            "color: #3fb950; font-size: 9px; font-weight: bold; "
            "font-family: 'Consolas', monospace;"
        )
        segment_count_label = QLabel()
        segment_count_label.setStyleSheet(
            "color: #6e7681; font-size: 9px; font-family: 'Consolas', monospace;"
        )
        header_layout.addWidget(live_indicator)
        header_layout.addStretch()
        header_layout.addWidget(segment_count_label)
        layout.addLayout(header_layout)

        text_area = QTextEdit()
        text_area.setReadOnly(True)
        text_area.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #c9d1d9;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #30363d;
            }
            QScrollBar:vertical {
                background-color: #161b22;
                width: 8px;
                border: none;
            }
            QScrollBar::handle:vertical {
                background-color: #30363d;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #484f58;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        bottom_bar = QHBoxLayout()
        auto_scroll_cb = QCheckBox("Auto-scroll")
        auto_scroll_cb.setChecked(True)
        auto_scroll_cb.setStyleSheet("""
            QCheckBox {
                color: #8b949e;
                font-size: 10px;
                spacing: 4px;
            }
            QCheckBox::indicator {
                width: 12px;
                height: 12px;
                border: 1px solid #30363d;
                border-radius: 2px;
                background-color: #21262d;
            }
            QCheckBox::indicator:checked {
                background-color: #238636;
                border-color: #2ea043;
            }
        """)
        copy_btn = QPushButton("📋 Copy Logs")
        copy_btn.setToolTip("Copy all segment history to clipboard as formatted text")
        copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #58a6ff;
                padding: 4px 10px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #1f3b5c;
                border-color: #58a6ff;
            }
        """)
        clear_btn = QPushButton("Clear History")
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #f85149;
                padding: 4px 10px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #3d1f1f;
                border-color: #f85149;
            }
        """)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #21262d;
                border: 1px solid #30363d;
                border-radius: 3px;
                color: #c9d1d9;
                padding: 4px 16px;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #30363d;
            }
        """)
        bottom_bar.addWidget(auto_scroll_cb)
        bottom_bar.addWidget(copy_btn)
        bottom_bar.addWidget(clear_btn)
        bottom_bar.addStretch()
        bottom_bar.addWidget(close_btn)
        layout.addLayout(bottom_bar)

        def render_log():
            """Build and display per-segment status history."""
            segments = []
            for seg_num in self._segment_order:
                if seg_num in self._segment_statuses:
                    segments.append(self._segment_statuses[seg_num])
            segment_count_label.setText(
                f"{len(segments)} segments (last {self._max_log_segments})"
            )
            if not segments:
                text_area.setHtml(
                    '<div style="color: #6e7681; padding: 30px; text-align: center;">'
                    "No segments processed yet<br>"
                    '<span style="font-size: 9px;">Start recording to see segment status</span>'
                    "</div>"
                )
                return
            html_parts = ['<div style="font-family: monospace;">']
            for seg_info in segments:
                seg_num = seg_info["segment_num"]
                duration = seg_info.get("duration")
                start_sec = seg_info.get("start_sec")
                retry_count = seg_info.get("retry_count", 0)
                final_status = seg_info.get("final_status")
                current_status = seg_info["current_status"]
                current_color = seg_info["current_color"]
                first_seen = seg_info["first_seen"]
                last_updated = seg_info["last_updated"]
                statuses = seg_info["statuses"]
                latest_info = statuses[-1].get("info", {}) if statuses else {}
                icon = self._get_status_icon(current_status, final_status, latest_info)
                color = self._get_status_color(final_status, current_color)
                duration_str = f" {duration:.1f}s" if duration else ""
                start_str = f" @{start_sec:.1f}s" if start_sec is not None else ""
                retry_str = f" [Retries: {retry_count}]" if retry_count > 0 else ""
                status_badge = ""
                if final_status == "success":
                    status_badge = (
                        '<span style="background:#1b3d1b; color:#3fb950; '
                        'font-size:8px; padding:1px 4px; border-radius:2px; margin-left:6px;">'
                        "SUCCESS</span>"
                    )
                elif final_status == "error":
                    status_badge = (
                        '<span style="background:#3d1f1f; color:#f85149; '
                        'font-size:8px; padding:1px 4px; border-radius:2px; margin-left:6px;">'
                        "ERROR</span>"
                    )
                header_html = (
                    f'<div style="'
                    f"background-color: #161b22; "
                    f"border-left: 3px solid {color}; "
                    f"padding: 6px 8px; "
                    f"margin-bottom: 4px; "
                    f"border-radius: 4px;"
                    f'">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f"<span>"
                    f'<span style="color: {color}; font-weight: bold; font-size: 12px;">'
                    f"{icon} Segment #{seg_num}{duration_str}{start_str}"
                    f"</span>"
                    f"{status_badge}"
                    f"</span>"
                    f'<span style="color: #6e7681; font-size: 9px;">'
                    f"{first_seen} → {last_updated}"
                    f"</span>"
                    f"</div>"
                )
                if retry_str:
                    header_html += (
                        f'<div style="margin-top: 3px;">'
                        f'<span style="color: #f0883e; font-size: 9px;">{retry_str}</span>'
                        f"</div>"
                    )
                if statuses:
                    display_statuses = (
                        statuses
                        if final_status
                        else (statuses[-3:] if len(statuses) > 3 else statuses)
                    )
                    header_html += (
                        '<div style="margin-top: 4px; font-size: 9px; color: #8b949e;">'
                    )
                    for status_entry in display_statuses:
                        entry_info = status_entry.get("info", {})
                        status_icon = self._get_status_icon(
                            status_entry["status"], None, entry_info
                        )
                        error_detail = ""
                        if entry_info.get("error"):
                            error_detail = (
                                f' <span style="color: #f85149;">'
                                f"({str(entry_info['error'])[:50]})"
                                f"</span>"
                            )
                        header_html += (
                            f'<div style="margin-left: 8px; opacity: 0.85;">'
                            f'<span style="color: #6e7681;">{status_entry["timestamp"]}</span> '
                            f"{status_icon} "
                            f'<span style="color: {status_entry["color"]};">'
                            f"{status_entry['status'][:80]}"
                            f"</span>"
                            f"{error_detail}"
                            f"</div>"
                        )
                    if not final_status and len(statuses) > 3:
                        header_html += (
                            f'<div style="margin-left: 8px; color: #6e7681; font-style: italic;">'
                            f"... and {len(statuses) - 3} more events"
                            f"</div>"
                        )
                    header_html += "</div>"
                header_html += "</div>"
                html_parts.append(header_html)
            html_parts.append("</div>")
            scrollbar = text_area.verticalScrollBar()
            was_at_bottom = scrollbar.value() >= scrollbar.maximum() - 10
            text_area.setHtml("".join(html_parts))
            if auto_scroll_cb.isChecked() and was_at_bottom:
                scrollbar.setValue(scrollbar.maximum())

        def copy_logs_to_clipboard():
            """Copy the plain text log to clipboard."""
            plain_text = self._generate_plain_text_log()
            QApplication.clipboard().setText(plain_text)
            original_text = copy_btn.text()
            copy_btn.setText("✓ Copied!")
            copy_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1b3d1b;
                    border: 1px solid #3fb950;
                    border-radius: 3px;
                    color: #3fb950;
                    padding: 4px 10px;
                    font-size: 10px;
                }
            """)
            QTimer.singleShot(1500, lambda: restore_copy_button(original_text))

        def restore_copy_button(original_text: str):
            """Restore copy button to original state."""
            copy_btn.setText(original_text)
            copy_btn.setStyleSheet("""
                QPushButton {
                    background-color: #21262d;
                    border: 1px solid #30363d;
                    border-radius: 3px;
                    color: #58a6ff;
                    padding: 4px 10px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background-color: #1f3b5c;
                    border-color: #58a6ff;
                }
            """)

        render_log()
        layout.insertWidget(1, text_area)

        live_timer = QTimer(dialog)
        live_timer.timeout.connect(render_log)
        live_timer.start(500)

        copy_btn.clicked.connect(copy_logs_to_clipboard)
        clear_btn.clicked.connect(lambda: self._clear_segment_history())
        close_btn.clicked.connect(dialog.close)

        def on_dialog_closed():
            live_timer.stop()
            if self._log_viewer is dialog:
                self._log_viewer = None
            console.print("[debug][LogHistory] Live segment viewer closed[/debug]")

        dialog.finished.connect(on_dialog_closed)

        # Close existing viewer if open
        if self._log_viewer is not None:
            self._log_viewer.close()
            self._log_viewer = None
        self._log_viewer = dialog
        dialog.show()
        console.print(
            "[debug][LogHistory] Live segment viewer opened — updating every 500ms[/debug]"
        )

    def _clear_segment_history(self) -> None:
        """Clear the segment status history."""
        self._segment_statuses.clear()
        self._segment_order.clear()
        console.print("[debug][SubtitleOverlay] Segment status history cleared[/debug]")
