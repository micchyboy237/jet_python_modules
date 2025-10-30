import builtins
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional
from jet.logger import logger
from jet.utils.inspect_utils import get_entry_file_name, inspect_original_script_path
from shared.setup.types import EventData


# ---- Event Settings ----
@dataclass
class _EventSettings:
    """EventSettings for event tracking."""

    _events: Dict[Literal["pre_start_hook", "post_start_hook"]
                  | str, EventData] = field(default_factory=dict)
    _event_data: EventData = field(default_factory=lambda: {
        "pre_start_hook": {"start_time": time.strftime('%Y-%m-%d|%H:%M:%S', time.gmtime())}
    })
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
        logger.debug(
            f"Setting event_data for current_event: {self.current_event}")
        self._event_data = event_data

        if not self.current_event:
            raise ValueError(
                'EventSettings.current_event must have one of ["pre_start_hook", "post_start_hook"]'
            )

        self.events[self.current_event] = event_data
        logger.debug(f"Updated events dictionary: {self.events}")

    def get_entry_event(self, event_name: Optional[str] = None) -> EventData:
        """Returns the event data for the specified event or the current event."""
        event_name = event_name or "pre_start_hook"
        if self.events.get(event_name):
            return self.events[event_name]
        raise ValueError("No current event set and no event_name provided")

    def get_entry_time(self, event_name: Optional[str] = None) -> str:
        """Returns the event data for the specified event or the current event."""
        event_name = event_name or "pre_start_hook"
        if self.events.get(event_name):
            return self.events[event_name]["start_time"]
        raise ValueError("No current event set and no event_name provided")

    def __getattr__(self, name):
        """Dynamically handle unknown event calls."""
        def _catch_all(*args, **kwargs) -> EventData:
            return self._catch_event_call(name, *args, **kwargs)
        return _catch_all

    def _catch_event_call(self, event_name: str, *args, **kwargs) -> EventData:
        """Handles all event calls dynamically."""
        logger.log("Event:", event_name, colors=["GRAY", "INFO"])
        self.current_event = event_name

        def format_callable(arg):
            return f"lambda_result=({arg()})" if callable(arg) else arg

        args = [format_callable(arg) for arg in args]
        kwargs = {key: format_callable(value) for key, value in kwargs.items()}

        result = inspect_original_script_path()
        event_data: EventData = {
            "event_name": event_name,
            "filename": get_entry_file_name(),
            "orig_function": result["last"],
            "arguments": {"args": args, "kwargs": kwargs},
            "start_time": time.strftime('%Y-%m-%d|%H:%M:%S', time.gmtime())
        }
        self.event_data[event_name] = event_data
        self.events[event_name] = event_data

        logger.log("File:", event_data.get(
            'filename', 'N/A'), colors=["GRAY", "ORANGE"])
        return event_data


_initialized = False
_initialize_lock = threading.Lock()


def setup_events():
    from jet.adapters.langchain.chat_agent_utils import reset_log_dir
    from jet.libs.llama_cpp.llamacpp_llm_interceptors import reset_llm_log_dir, setup_llamacpp_llm_interceptors
    from jet.libs.llama_cpp.llamacpp_embed_interceptors import reset_embed_log_dir, setup_llamacpp_embed_interceptors

    reset_log_dir() # Reset logs
    reset_embed_log_dir()
    reset_llm_log_dir()

    setup_llamacpp_llm_interceptors()
    setup_llamacpp_embed_interceptors()

    global _initialized
    if _initialized:
        return

    with _initialize_lock:
        if _initialized:
            return
        from shared.setup.events import EventSettings

        def pre_start_hook():
            event_data = EventSettings.pre_start_hook()
            logger.newline()
            logger.teal("pre_start_hook triggered at: " +
                           EventSettings.get_entry_time())

        pre_start_hook()

        _initialized = True


# Singleton
if not hasattr(builtins, "EventSettings"):
    EventSettings = _EventSettings()
    builtins.EventSettings = EventSettings
EventSettings = builtins.EventSettings


__all__ = [
    "EventSettings",
]

if __name__ == "__main__":
    logger.newline()
    logger.info("Run 1")
    setup_events()
    logger.newline()
    logger.info("Run 2")
    setup_events()
    logger.newline()
    logger.info("Run 3")
    setup_events()
