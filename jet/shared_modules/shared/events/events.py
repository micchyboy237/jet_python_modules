import builtins

import time
import threading

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Literal, Optional
from jet.logger.config import configure_logger
from shared.globals import EventData
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import inspect_original_script_path
import logging
from shared.globals import import_tracker


# ---- Custom Logger ----
# class RefreshableLoggerHandler(logging.Logger):
#     def __init__(self, name: str):
#         super().__init__(name)
#         self.max_wait_time = 5  # Default wait time
#         self.check_interval = 1  # Check every 1 second

#     def __getattr__(self, name: str):
#         self.refresh_wait_time()

#     def refresh_wait_time(self):
#         """Refresh the max wait time whenever a logging call is made."""
#         global import_tracker
#         import_tracker.refresh_wait_time(self.max_wait_time)


class RefreshableLoggerHandler(logging.Handler):
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, record):
        # Here we handle the log record and define custom behavior for each log level
        log_message = self.format(record)

        # Listen to any log level and perform custom actions
        print(f"[Listener] Log received: {log_message}")

        # Refresh the max wait time whenever a logging call is made.
        import_tracker.refresh_wait_time()

    # def __getattr__(self, name: str):
    #     self.refresh_wait_time()


# ---- Event Settings ----
@dataclass
class _EventSettings:
    """EventSettings for event tracking."""

    _events: Dict[Literal["pre_start_hook", "post_start_hook"]
                  | str, EventData] = field(default_factory=dict)
    _event_data: EventData = field(default_factory=dict)
    _current_event: Optional[Literal["pre_start_hook",
                                     "post_start_hook"] | str] = None

    @property
    def events(self) -> Dict[Literal["pre_start_hook", "post_start_hook"] | str, EventData]:
        return self._events

    @events.setter
    def events(self, events: Dict[Literal["pre_start_hook", "post_start_hook"] | str, EventData]) -> None:
        self._events = events

    @property
    def current_event(self) -> Optional[Literal["pre_start_hook", "post_start_hook"] | str]:
        return self._current_event

    @current_event.setter
    def current_event(self, current_event: Literal["pre_start_hook", "post_start_hook"] | str) -> None:
        self._current_event = current_event

    @property
    def event_data(self) -> EventData:
        return self._event_data

    @event_data.setter
    def event_data(self, event_data: EventData) -> None:
        self._event_data = event_data

        if not self.current_event:
            raise ValueError(
                'EventSettings.current_event must have one of ["pre_start_hook", "post_start_hook"]'
            )

        self.events[self.current_event] = event_data

    def __getattr__(self, name):
        """Dynamically handle unknown event calls."""
        def _catch_all(*args, **kwargs) -> EventData:
            return self._catch_event_call(name, *args, **kwargs)
        return _catch_all

    def _catch_event_call(self, event_name: str, *args, **kwargs) -> EventData:
        """Handles all event calls dynamically."""
        logger.orange(f"Event: {event_name}")
        EventSettings.current_event = event_name

        def format_callable(arg):
            return f"lambda_result=({arg()})" if callable(arg) else arg

        args = [format_callable(arg) for arg in args]
        kwargs = {key: format_callable(value) for key, value in kwargs.items()}

        result = inspect_original_script_path()
        event_data: EventData = {
            "event_name": event_name,
            **(result["first"] if result.get("first") else {}),
            "orig_function": result["last"],
            "arguments": {"args": args, "kwargs": kwargs},
            "start_time": time.strftime('%Y-%m-%d|%H:%M:%S', time.gmtime())
        }
        EventSettings.event_data[event_name] = event_data

        logger.log(f"File:", event_data.get(
            'filename', 'N/A'), colors=["GRAY", "ORANGE"])
        return event_data


_initialized = False
_initialize_lock = threading.Lock()


def setup_events():
    global _initialized
    if _initialized:
        return

    with _initialize_lock:
        if _initialized:
            return
        from shared.events import EventSettings

        def pre_start_hook():
            # Initialize base logger
            EventSettings.pre_start_hook(configure_logger)
            logger.newline()
            logger.success("pre_start_hook triggered at: " +
                           EventSettings.event_data['pre_start_hook']['start_time'])

        def post_start_hook():
            # Now trigger post_start_hook
            EventSettings.post_start_hook()
            logger.newline()
            logger.success("post_start_hook triggered at: " +
                           EventSettings.event_data['post_start_hook']['start_time'])

        # Explicitly wait for the post_start_event before proceeding with a max wait time of 5 seconds
        import_tracker.wait_for_all_modules({
            "pre_start_hook": pre_start_hook,
            "post_start_hook": post_start_hook,
        })

        _initialized = True


# Singleton
if not hasattr(builtins, "EventSettings"):
    EventSettings = _EventSettings()
    builtins.EventSettings = EventSettings
EventSettings = builtins.EventSettings


__all__ = [
    "EventSettings",
    "initialize"
]

if __name__ == "__main__":
    logger.newline()
    logger.info("Run 1")
    setup_events()  # Runs the initialization process
    logger.newline()
    logger.info("Run 2")
    setup_events()  # Does nothing
    logger.newline()
    logger.info("Run 3")
    setup_events()  # Still does nothing
