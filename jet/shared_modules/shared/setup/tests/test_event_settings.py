import pytest
import re
from unittest.mock import patch
from typing import Dict, Optional
from jet.logger import logger
from jet.utils.inspect_utils import get_entry_file_name, inspect_original_script_path
from shared.setup.types import EventData
from shared.setup.events import _EventSettings, EventSettings


@pytest.fixture
def event_settings():
    """Fixture to create a fresh _EventSettings instance."""
    return _EventSettings()


class TestEventSettings:
    """Tests for _EventSettings class behavior."""

    def test_get_entry_event_with_current_event(self, event_settings):
        """Test retrieving event data for the current event."""
        # Given: An event settings instance with a current event and event data
        event_settings.current_event = "pre_start_hook"
        expected_data: EventData = {
            "event_name": "pre_start_hook",
            "filename": "test_file.py",
            "orig_function": "test_function",
            "arguments": {"args": [], "kwargs": {}},
            "start_time": "2025-09-08|01:37:00"
        }
        event_settings.events["pre_start_hook"] = expected_data

        # When: We call get_entry_event without specifying an event name
        with patch("time.strftime", return_value="2025-09-08|01:37:00"):
            result = event_settings.get_entry_event()

        # Then: The event data for the current event is returned
        assert result == expected_data, f"Expected {expected_data}, but got {result}"

    def test_get_entry_event_with_specific_event(self, event_settings):
        """Test retrieving event data for a specific event name."""
        # Given: An event settings instance with event data
        expected_data: EventData = {
            "event_name": "post_start_hook",
            "filename": "test_file.py",
            "orig_function": "test_function",
            "arguments": {"args": [], "kwargs": {}},
            "start_time": "2025-09-08|01:37:00"
        }
        event_settings.events["post_start_hook"] = expected_data

        # When: We call get_entry_event with a specific event name
        result = event_settings.get_entry_event("post_start_hook")

        # Then: The event data for the specified event is returned
        assert result == expected_data, f"Expected {expected_data}, but got {result}"

    def test_get_entry_event_no_current_event(self, event_settings):
        """Test get_entry_event raises ValueError when no current event is set."""
        # Given: An event settings instance with no current event
        event_settings.current_event = None

        # When: We call get_entry_event without an event name
        # Then: A ValueError is raised
        with pytest.raises(ValueError, match="No current event set and no event_name provided"):
            event_settings.get_entry_event()

    def test_dynamic_event_call(self, event_settings):
        """Test dynamic event call sets event data correctly."""
        # Given: An event settings instance and mocked utilities
        with patch("jet.utils.inspect_utils.get_entry_file_name", return_value="test_event_settings"), \
                patch("jet.utils.inspect_utils.inspect_original_script_path", return_value={"last": "__main__"}), \
                patch("time.strftime", return_value="2025-09-08|01:37:00"):
            # When: We trigger a dynamic event call
            result = event_settings.custom_event("arg1", key="value")

            # Then: The event data is correctly set and returned
            expected_data: EventData = {
                "event_name": "custom_event",
                "filename": "__main__.py",
                "orig_function": {
                    "filepath": "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/shared_modules/shared/setup/tests/test_event_settings.py",
                    "filename": "test_event_settings.py",
                    "function": "test_dynamic_event_call",
                    "lineno": 75,
                    "code_context": [
                        "            result = event_settings.custom_event(\"arg1\", key=\"value\")\n"
                    ]
                },
                "arguments": {
                    "args": [
                        "arg1"
                    ],
                    "kwargs": {
                        "key": "value"
                    }
                },
                "start_time": "2025-09-08|01:37:00"
            }
            assert result == expected_data, f"Expected {expected_data}, but got {result}"
            assert EventSettings.get_entry_event() == expected_data, \
                f"Expected {expected_data}, but got {result}"
            assert event_settings.events["custom_event"] == expected_data, \
                f"Expected event data {expected_data}, but got {event_settings.events['custom_event']}"

    def test_event_data_setter(self, event_settings):
        """Test setting event data updates events dictionary."""
        # Given: An event settings instance with a current event
        event_settings.current_event = "pre_start_hook"
        new_event_data: EventData = {
            "event_name": "pre_start_hook",
            "filename": "script.py",
            "orig_function": "start_function",
            "arguments": {"args": ["test"], "kwargs": {"param": "value"}},
            "start_time": "2025-09-08|01:37:00"
        }

        # When: We set the event_data property
        event_settings.event_data = new_event_data

        # Then: The events dictionary is updated with the new event data
        assert event_settings.events["pre_start_hook"] == new_event_data, \
            f"Expected {new_event_data}, but got {event_settings.events['pre_start_hook']}"

    def test_event_data_setter_no_current_event(self, event_settings):
        """Test setting event_data without current_event raises ValueError."""
        # Given: An event settings instance with no current event
        event_settings.current_event = None
        new_event_data: EventData = {
            "event_name": "pre_start_hook",
            "filename": "script.py",
            "orig_function": "start_function",
            "arguments": {"args": [], "kwargs": {}},
            "start_time": "2025-09-08|01:37:00"
        }

        # When: We try to set event_data
        # Then: A ValueError is raised
        with pytest.raises(ValueError, match=re.escape('EventSettings.current_event must have one of ["pre_start_hook", "post_start_hook"]')):
            event_settings.event_data = new_event_data
