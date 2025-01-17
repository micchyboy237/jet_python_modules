from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional
from global_types import EventData
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

    def _catch_event_call(self, event_name: str, *args,  **kwargs) -> 'EventData':
        # Add logic here for handling all calls
        """
        Runs the `inspect_original_script_path` function and stores the result in the event_data attribute.
        """
        logger.info(f"Event: {event_name}")
        self.current_event = event_name

        result = inspect_original_script_path()
        event_data: 'EventData' = {
            "event_name": event_name,
            **(result["first"] if result.get("first") else {}),
            "orig_function": result["last"],
            "arguments": {"args": args, "kwargs": kwargs}
        }
        self.event_data = event_data

        # To verify, print the result
        logger.log(f"File:", event_data['filename'], colors=["GRAY", "ORANGE"])
        # logger.orange(format_json(EventSettings.event_data)[:100])
        return event_data

    def __getattr__(self, name):
        # Return a function that delegates to `all_catch_call`
        def _catch_all(*args, **kwargs) -> 'EventData':

            return self._catch_event_call(name, *args,  **kwargs)
        return _catch_all


# Singleton
EventSettings = _EventSettings()


if __name__ == "__main__":
    logger.newline()
    logger.info("Event 1...")
    event = EventSettings.any_call1(1, 2)
    logger.success(format_json(event))

    logger.newline()
    logger.info("Event 2...")
    event = EventSettings.any_call2("test", model="model")
    logger.success(format_json(event))
    pass
