from datetime import datetime
from typing import Optional


class TimeTracker:
    """A singleton class to track the start and end times of a process."""

    _instance: Optional['TimeTracker'] = None

    def __new__(cls) -> 'TimeTracker':
        """Ensures only one instance of TimeTracker is created."""
        if cls._instance is None:
            cls._instance = super(TimeTracker, cls).__new__(cls)
            cls._instance._start_time = None
            cls._instance._end_time = None
        return cls._instance

    def start_process(self) -> None:
        """Records the start time of the process."""
        self._start_time = datetime.now()

    def stop_process(self) -> None:
        """Records the end time of the process."""
        self._end_time = datetime.now()

    @property
    def start_time(self) -> Optional[datetime]:
        """Returns the start time of the process."""
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        """Returns the end time of the process."""
        return self._end_time

    @property
    def duration(self) -> Optional[float]:
        """Returns the duration of the process in seconds, or None if not started."""
        if self._start_time:
            end_time = self._end_time if self._end_time else datetime.now()
            return (end_time - self._start_time).total_seconds()
        return None

    @staticmethod
    def start() -> None:
        """Records the start time of the singleton instance."""
        TimeTracker().start_process()

    @staticmethod
    def stop() -> None:
        """Records the end time of the singleton instance."""
        TimeTracker().stop_process()

    @staticmethod
    def get_start_time() -> Optional[datetime]:
        """Returns the start time of the singleton instance."""
        return TimeTracker().start_time

    @staticmethod
    def get_end_time() -> Optional[datetime]:
        """Returns the end time of the singleton instance."""
        return TimeTracker().end_time

    @staticmethod
    def get_duration() -> Optional[float]:
        """Returns the duration of the singleton instance in seconds."""
        return TimeTracker().duration
