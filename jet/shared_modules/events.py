import inspect
import os
from typing import Optional, Dict

from global_types import EventData
from jet.logger import logger
from jet.transformers.formatters import format_json
from jet.utils.inspect_utils import inspect_original_script_path


class _EventSettings:
    event_data: Optional[EventData] = None
    _instance: Optional['_EventSettings'] = None

    def __new__(cls):
        # Ensure only one instance exists
        if cls._instance is None:
            cls._instance = super(_EventSettings, cls).__new__(cls)
        return cls._instance

    def pre_start_hook(self):
        """
        Runs the `inspect_original_script_path` function and stores the result in the event_data attribute.
        """
        logger.info("Event: pre_start_hook")

        result = inspect_original_script_path()
        self.event_data = result

        # To verify, print the result
        if self.event_data:
            logger.debug(f"EventSettings.event_data:")
            logger.success(format_json(self.event_data))
        else:
            logger.error("No event data found")

    def get_event_data(self) -> Optional[EventData]:
        """Returns the event data stored in the singleton."""
        return self.event_data


# Create the singleton instance of EventSettings
EventSettings = _EventSettings()


if __name__ == "__main__":
    pass
