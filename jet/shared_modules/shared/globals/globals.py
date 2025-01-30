import builtins
import json
import logging
import sys
import importlib
import threading
import time
from typing import Callable, Optional, TypedDict

from jet.logger.timer import SleepInterruptible, SleepStatus, sleep_countdown
from jet.utils.inspect_utils import find_stack_frames, get_stack_frames, print_inspect_original_script_path

# ANSI color codes
BOLD = "\u001b[1m"
RESET = "\u001b[0m"
COLORS = {
    "INFO": BOLD + "\u001b[38;5;213m",
    "BRIGHT_INFO": BOLD + "\u001b[48;5;213m",
    "DEBUG": BOLD + "\u001b[38;5;45m",
    "BRIGHT_DEBUG": BOLD + "\u001b[48;5;45m",
    "WARNING": BOLD + "\u001b[38;5;220m",
    "BRIGHT_WARNING": BOLD + "\u001b[48;5;220m",
    "ERROR": BOLD + "\u001b[38;5;196m",
    "BRIGHT_ERROR": BOLD + "\u001b[48;5;196m",
    "CRITICAL": BOLD + "\u001b[38;5;124m",
    "BRIGHT_CRITICAL": BOLD + "\u001b[48;5;124m",
    "SUCCESS": BOLD + "\u001b[38;5;40m",
    "BRIGHT_SUCCESS": BOLD + "\u001b[48;5;40m",
    "ORANGE": BOLD + "\u001b[38;5;208m",
    "BRIGHT_ORANGE": BOLD + "\u001b[48;5;208m",
    "PURPLE": BOLD + "\u001b[38;5;92m",
    "BRIGHT_PURPLE": BOLD + "\u001b[48;5;92m",
    "LIME": BOLD + "\u001b[38;5;82m",
    "BRIGHT_LIME": BOLD + "\u001b[48;5;82m",
    "WHITE": BOLD + "\u001b[38;5;15m",
    "GRAY": "\u001b[38;5;250m",
    "LOG": BOLD + "\u001b[38;5;15m",
    "RESET": "\u001b[0m",
}


def colorize_log(text: str, color: str):
    return f"{color}{text}{RESET}"


# ---- Custom Logger ----
class RefreshableLoggerHandler(logging.Handler):
    def emit(self, record):
        """Refresh the max wait time whenever a logging call is made."""

        if record and hasattr(record, "message"):
            frame = find_stack_frames(record.message)
            if frame:
                global import_tracker
                import_tracker.refresh_wait_time()

                # print("FRAME:")
                # print(json.dumps(frame, indent=2))


# ---- Initial Pre Post Hooks ----
class PrePostHooks(TypedDict, total=False):
    pre_start_hook: Callable
    post_start_hook: Callable


# ---- Comprehensive Import Tracker ----
class ImportTracker:
    """
    Tracks all module imports, including dynamic ones.
    """

    def __init__(self):
        # Start with already loaded modules
        self.tracked_modules = set(sys.modules.keys())
        self.lock = threading.Lock()
        self.post_start_event = threading.Event()  # Synchronization flag
        self.running = True
        self.max_wait_time = 10  # Default max wait time
        self.check_interval = 1  # Check every 1 second
        self.elapsed_time = 0
        self.sleep_manager = SleepInterruptible(
            self.max_wait_time, on_complete=self._is_loading_complete)

        self.start_hook: Optional[Callable] = None
        self.end_hook: Optional[Callable] = None

    def _is_loading_complete(self, status: SleepStatus, total_elapsed: float, restart_elapsed: float):
        self.post_start_event.set()

        print()
        print(
            colorize_log(
                "Loaded Status:",
                COLORS["WHITE"]
            ),
            colorize_log(
                status,
                COLORS["DEBUG"]
            ),
            colorize_log(
                "|",
                COLORS["GRAY"]
            ),
            colorize_log(
                "Slept:",
                COLORS["WHITE"]
            ),
            colorize_log(
                f"{total_elapsed:.2f}s",
                COLORS["DEBUG"]
            ),
            colorize_log(
                "|",
                COLORS["GRAY"]
            ),
            colorize_log(
                "Restarted:",
                COLORS["WHITE"]
            ),
            colorize_log(
                f"{restart_elapsed:.2f}s",
                COLORS["DEBUG"]
            ),
        )
        print(colorize_log(
            "Done loading all modules!",
            COLORS["SUCCESS"]
        ))

        if self.end_hook:
            self.end_hook()

    def register_module(self, module_name: str):
        """Register a module that is expected to load."""
        with self.lock:
            self.tracked_modules.add(module_name)

    def mark_module_loaded(self, module_name: str):
        """Mark a module as loaded and check completion."""
        with self.lock:
            self.tracked_modules.add(module_name)

    def has_loaded_system_modules(self) -> bool:
        """Check if new modules are still loading."""
        time.sleep(self.check_interval)  # Allow modules to load in parallel
        with self.lock:
            # return len(sys.modules.keys()) == len(self.tracked_modules)
            return bool(len(self.tracked_modules))

    def wait_for_all_modules(self, options: Optional[PrePostHooks] = None):
        """Monitor until all imports settle, with a max wait time."""
        if options:
            self.start_hook = options.get("pre_start_hook")
            self.end_hook = options.get("post_start_hook")

        if self.start_hook:
            self.start_hook()

        print(colorize_log(
            "Waiting for all modules to load...",
            COLORS["DEBUG"]
        ))
        while self.running:
            loaded_system_modules = self.has_loaded_system_modules()
            time.sleep(self.check_interval)

            if loaded_system_modules:
                self.sleep_manager.start_sleep()
                break

    def refresh_wait_time(self):
        """Refresh the max wait time when a log is called."""
        self.sleep_manager.restart_sleep()


# ---- Hook `importlib.import_module()` ----
original_import_module = importlib.import_module


def custom_import_module(name, package=None):
    """Custom import hook to track dynamically imported modules."""
    module = original_import_module(name, package)
    import_tracker.mark_module_loaded(name)
    return module


importlib.import_module = custom_import_module  # Monkey-patch importlib


# ---- Start Import Monitoring Thread ----
import_tracker = ImportTracker()
import_monitor_thread = threading.Thread(
    target=import_tracker.wait_for_all_modules, daemon=True)
import_monitor_thread.start()


__all__ = [
    "RefreshableLoggerHandler",
    "import_tracker",
]
