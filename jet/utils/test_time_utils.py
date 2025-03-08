import unittest

from jet.utils.time_utils import format_time


class TestFormatTime(unittest.TestCase):

    def test_seconds_only(self):
        # Less than 1 minute, should return seconds only
        self.assertEqual(format_time(59), "59s")

    def test_minutes_and_seconds(self):
        # Between 1 minute and 1 hour, should return minutes and seconds
        self.assertEqual(format_time(90), "1m 30s")

    def test_hours_minutes_seconds(self):
        # More than 1 hour, should return hours, minutes, and seconds
        self.assertEqual(format_time(3661), "1h 1m 1s")

    def test_exactly_one_hour(self):
        # Exactly 1 hour, should return hours and minutes
        self.assertEqual(format_time(3600), "1h")

    def test_exactly_one_minute(self):
        # Exactly 1 minute, should return minutes and seconds
        self.assertEqual(format_time(60), "1m")

    def test_no_duration(self):
        # Zero duration, should return 0 seconds
        self.assertEqual(format_time(0), "0s")

    def test_fractional_seconds(self):
        # Fractional seconds should still be rounded and formatted
        self.assertEqual(format_time(59.99), "59s")
        self.assertEqual(format_time(90.1), "1m 30s")


if __name__ == "__main__":
    unittest.main()
