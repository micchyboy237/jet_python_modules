import pytest
from datetime import datetime
from shared.time_tracker import TimeTracker


class TestTimeTracker:
    def test_singleton_instance(self):
        # Given: Two attempts to create TimeTracker instances
        tracker1 = TimeTracker()
        tracker2 = TimeTracker()

        # When: We compare the instances
        result = tracker1 is tracker2

        # Then: They should be the same instance
        expected = True
        assert result == expected, "TimeTracker should return the same instance"

    def test_start_records_current_time(self):
        # Given: A new TimeTracker instance
        tracker = TimeTracker()
        expected_time_before_start = datetime.now()

        # When: The start method is called
        tracker.start()

        # Then: The start_time is set to a recent datetime
        result = tracker.start_time
        assert isinstance(result, datetime)
        # Allow small time delta
        assert (result - expected_time_before_start).total_seconds() < 1

    def test_stop_records_current_time(self):
        # Given: A TimeTracker instance with a start time
        tracker = TimeTracker()
        tracker.start()
        expected_time_before_stop = datetime.now()

        # When: The stop method is called
        tracker.stop()

        # Then: The end_time is set to a recent datetime
        result = tracker.end_time
        assert isinstance(result, datetime)
        # Allow small time delta
        assert (result - expected_time_before_stop).total_seconds() < 1

    def test_duration_calculates_time_difference(self):
        # Given: A TimeTracker instance with start and stop times
        tracker = TimeTracker()
        tracker.start()
        import time
        time.sleep(0.5)  # Simulate some work

        # When: The stop method is called
        tracker.stop()

        # Then: The duration is calculated correctly
        result = tracker.duration
        expected = 0.5  # Approximately 0.5 seconds
        assert isinstance(result, float)
        assert abs(result - expected) < 0.1  # Allow small variance due to system timing

    def test_duration_returns_none_if_not_started(self):
        # Given: A TimeTracker instance that has not started
        tracker = TimeTracker()

        # When: Duration is accessed
        result = tracker.duration

        # Then: Duration is None
        expected = None
        assert result == expected, "Duration should be None when process has not started"

    def test_duration_calculates_with_no_stop(self):
        # Given: A TimeTracker instance that has started but not stopped
        tracker = TimeTracker()
        tracker.start()
        import time
        time.sleep(0.5)

        # When: Getting the duration
        result = tracker.duration

        # Then: Duration should be approximately 0.5 seconds
        expected = 0.5
        assert isinstance(result, float)
        assert abs(result - expected) < 0.1

    def test_static_get_duration_with_no_stop(self):
        # Given: A TimeTracker singleton instance that has started but not stopped
        TimeTracker.start()
        import time
        time.sleep(0.5)

        # When: Getting the duration through static method
        result = TimeTracker.get_duration()

        # Then: Duration should be approximately 0.5 seconds
        expected = 0.5
        assert isinstance(result, float)
        assert abs(result - expected) < 0.1

    def test_static_get_start_time(self):
        # Given: A TimeTracker instance with a start time
        tracker = TimeTracker()
        tracker.start()
        expected = tracker.start_time

        # When: The static get_start_time method is called
        result = TimeTracker.get_start_time()

        # Then: The result matches the instance start_time
        assert result == expected
        assert isinstance(result, datetime)

    def test_static_get_end_time(self):
        # Given: A TimeTracker instance with start and stop times
        tracker = TimeTracker()
        tracker.start()
        tracker.stop()
        expected = tracker.end_time

        # When: The static get_end_time method is called
        result = TimeTracker.get_end_time()

        # Then: The result matches the instance end_time
        assert result == expected
        assert isinstance(result, datetime)

    def test_static_get_duration(self):
        # Given: A TimeTracker instance with start and stop times
        tracker = TimeTracker()
        tracker.start()
        import time
        time.sleep(0.5)  # Simulate some work
        tracker.stop()
        expected = tracker.duration

        # When: The static get_duration method is called
        result = TimeTracker.get_duration()

        # Then: The result matches the instance duration
        assert result == expected
        assert isinstance(result, float)
        assert abs(result - 0.5) < 0.1  # Allow small variance

    def test_static_start(self):
        # Given: A TimeTracker instance
        expected_time_before_start = datetime.now()

        # When: The static start method is called
        TimeTracker.start()

        # Then: The start_time is set to a recent datetime
        result = TimeTracker.get_start_time()
        assert isinstance(result, datetime)
        # Allow small time delta
        assert (result - expected_time_before_start).total_seconds() < 1

    def test_static_stop(self):
        # Given: A TimeTracker instance with a start time
        TimeTracker.start()
        expected_time_before_stop = datetime.now()

        # When: The static stop method is called
        TimeTracker.stop()

        # Then: The end_time is set to a recent datetime
        result = TimeTracker.get_end_time()
        assert isinstance(result, datetime)
        # Allow small time delta
        assert (result - expected_time_before_stop).total_seconds() < 1


@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Given: Reset the singleton instance before each test to avoid state leakage
    TimeTracker._instance = None
    yield
