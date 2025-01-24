import builtins
from dataclasses import dataclass, field
import time
from typing import Any, Callable, List, Literal, Optional
from shared.global_types import EventData
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import inspect_original_script_path


@dataclass
class _EventSettings:
    """EventSettings for the Llama Index, lazily initialized."""

    # lazy initialization
    _events: dict[Literal["pre_start_hook"] | str,
                  EventData] = field(default_factory=dict)
    _event_data: 'EventData' = field(default_factory=dict)
    _current_event: Optional[Literal["pre_start_hook"] | str] = None

    # ---- Properties ----

    @property
    def events(self) -> 'dict[Literal["pre_start_hook"] | str, EventData] ':
        """Get the event data."""
        return self._events

    @events.setter
    def events(self, events: 'dict[Literal["pre_start_hook"] | str, EventData] ') -> None:
        """Set the events."""
        self._events = events

    @property
    def current_event(self) -> Optional['Literal["pre_start_hook"] | str']:
        """Get the current_event data."""
        return self._current_event

    @current_event.setter
    def current_event(self, current_event: 'Literal["pre_start_hook"] | str') -> None:
        """Set the current_event."""
        self._current_event = current_event

    @property
    def event_data(self) -> 'EventData':
        """Get the event data."""
        return self._event_data

    @event_data.setter
    def event_data(self, event_data: 'EventData') -> None:
        """Set the event data."""
        self._event_data = event_data

        if not self.current_event:
            raise ValueError(
                "EventSettings.current_event must have one of [\"pre_start_hook\"]")

        self.events[self.current_event] = event_data

    # ---- Events ----
    def __getattr__(self, name):
        # Return a function that delegates to `all_catch_call`
        def _catch_all(*args, **kwargs) -> 'EventData':
            return self._catch_event_call(name, *args,  **kwargs)
        return _catch_all

    def _catch_event_call(self, event_name: str, *args,  **kwargs) -> 'EventData':
        # Add logic here for handling all calls
        """
        Runs the `inspect_original_script_path` function and stores the result in the event_data attribute.
        """
        logger.orange(f"Event: {event_name}")
        EventSettings.current_event = event_name

        # Execute callable arguments and format results
        def format_callable(arg):
            if callable(arg):
                return f"lambda_result=({arg()})"
            return arg

        args = [format_callable(arg) for arg in args]
        kwargs = {key: format_callable(value) for key, value in kwargs.items()}

        result = inspect_original_script_path()
        event_data: 'EventData' = {
            "event_name": event_name,
            **(result["first"] if result.get("first") else {}),
            "orig_function": result["last"],
            "arguments": {"args": args, "kwargs": kwargs},
            "start_time": time.strftime('%Y-%m-%d|%H:%M:%S', time.gmtime())
        }
        EventSettings.event_data[event_name] = event_data

        # To verify, print the result
        logger.log(f"File:", event_data['filename'], colors=["GRAY", "ORANGE"])
        return event_data


# Singleton
if not hasattr(builtins, "EventSettings"):
    EventSettings = _EventSettings()
    builtins.EventSettings = EventSettings
EventSettings = builtins.EventSettings

__all__ = [
    "EventSettings"
]

if __name__ == "__main__":
    from jet.logger.config import configure_logger
    from shared.events import EventSettings

    # Trigger event event_pre_start_hook
    EventSettings.pre_start_hook(configure_logger)

    logger.newline()
    pre_start_hook_start_time = EventSettings.event_data["pre_start_hook"]["start_time"]
    logger.info("pre_start_hook start time:")
    logger.success(pre_start_hook_start_time)

    logger.newline()
    event = EventSettings.any_call1(lambda: 1 + 2, sample_func=lambda: 2 + 3)
    logger.info("Event 1...")
    logger.success(format_json(event))

    logger.newline()
    event = EventSettings.any_call2(
        "test", sample_func=lambda: "dynamic_model")
    logger.info("Event 2...")
    logger.success(format_json(event))
    pass
