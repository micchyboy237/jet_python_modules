import rumps
import psutil
import sys
from PyQt6.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt


class RAMMonitorWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAM Monitor")
        self.setGeometry(100, 100, 400, 200)  # Adjust size
        # Keep on top of other windows
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        layout = QVBoxLayout()

        # RAM and processes labels
        self.ram_label = QLabel(self.get_ram_usage(), self)
        self.process_label = QLabel(self.get_top_processes(), self)

        # Refresh and Quit buttons
        self.refresh_button = QPushButton('Refresh', self)
        self.quit_button = QPushButton('Quit', self)

        self.refresh_button.clicked.connect(self.refresh)
        self.quit_button.clicked.connect(self.quit_app)

        # Add widgets to layout
        layout.addWidget(self.ram_label)
        layout.addWidget(self.process_label)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.quit_button)

        self.setLayout(layout)

    def update_data(self):
        """Update the RAM usage and process list dynamically."""
        self.ram_label.setText(self.get_ram_usage())
        self.process_label.setText(self.get_top_processes())

    def get_ram_usage(self):
        mem = psutil.virtual_memory()
        return f'RAM Usage: {mem.used / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent:.2f}%)'

    def get_top_processes(self):
        processes = []
        for p in psutil.process_iter(['name', 'memory_info']):
            try:
                if p.info['memory_info']:
                    processes.append(
                        (p.info['memory_info'].rss, p.info['name']))
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass

        processes.sort(reverse=True, key=lambda x: x[0])
        top_processes = processes[:5]
        return "\n".join([f"{name}: {mem / (1024**3):.2f} GB" for mem, name in top_processes])

    def refresh(self):
        """Manually refresh RAM usage in the window."""
        self.update_data()

    def quit_app(self):
        """Close the app and exit cleanly."""
        self.close()
        sys.exit()


class RAMMonitorApp(rumps.App):
    def __init__(self):
        super().__init__("RAM Monitor", icon=None)
        self.window = None
        self.app = None  # PyQt application reference
        self.menu = []  # No dropdown menu here, just the icon

        # Create QApplication before initializing the window
        self.app = QApplication(sys.argv)

        # Show the window immediately on startup
        self.window = RAMMonitorWindow()
        self.window.show()

        # Update the RAM usage icon on startup
        self.update_ram_usage()

    def update_ram_usage(self):
        """Update the menu bar title with RAM usage."""
        mem = psutil.virtual_memory()
        usage_percentage = f'{mem.percent:.1f}%'
        self.icon = None  # Set an empty icon if you don't want to use a custom one
        self.title = f'RAM: {usage_percentage}'
        # Update window with fresh RAM data
        self.window.ram_label.setText(self.window.get_ram_usage())

        # Update the RAM usage in the menu bar
        self.icon = None  # You can use an icon if you want, or keep it empty
        # Title is shown on the menu bar with RAM usage
        self.title = f'RAM: {usage_percentage}'

    @rumps.clicked("RAM Monitor")
    def toggle_window(self, _):
        """Toggle the visibility of the RAM usage window."""
        if self.window.isVisible():
            self.window.hide()
        else:
            self.window.update_data()
            self.window.show()


if __name__ == "__main__":
    app = RAMMonitorApp()
    app.run()
