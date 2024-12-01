import unittest
import tempfile
import os
from datetime import datetime, timedelta
from jet.time.utils import get_file_dates, calculate_time_difference, readable_time_difference


class TestUtilityFunctions(unittest.TestCase):

    def test_get_file_dates(self):
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_name = temp_file.name

        try:
            # Get file dates as datetime objects
            result = get_file_dates(temp_file_name)
            self.assertIn("created", result)
            self.assertIn("modified", result)
            self.assertIsInstance(result["created"], datetime)
            self.assertIsInstance(result["modified"], datetime)

            # Get file dates as formatted strings
            format_str = "%Y-%m-%d %H:%M:%S"
            formatted_result = get_file_dates(
                temp_file_name, format=format_str)
            self.assertIsInstance(formatted_result["created"], str)
            self.assertIsInstance(formatted_result["modified"], str)
            self.assertEqual(
                formatted_result["created"],
                result["created"].strftime(format_str)
            )
            self.assertEqual(
                formatted_result["modified"],
                result["modified"].strftime(format_str)
            )
        finally:
            # Clean up temporary file
            os.remove(temp_file_name)

    def test_calculate_time_difference(self):
        # Set a past time
        past_time = datetime.now() - timedelta(days=1, hours=2, minutes=30)
        time_diff = calculate_time_difference(past_time)

        self.assertGreater(time_diff, timedelta(days=1))
        self.assertLess(time_diff, timedelta(days=2))

    def test_readable_time_difference(self):
        # Test cases with known timedelta values
        test_cases = [
            (timedelta(days=2, hours=3), "2 days, 3 hours"),
            (timedelta(hours=5, minutes=45), "5 hours, 45 minutes"),
            (timedelta(minutes=15, seconds=30), "15 minutes, 30 seconds"),
            (timedelta(seconds=10), "10 seconds"),
            (timedelta(), "just now")
        ]

        for td, expected in test_cases:
            self.assertEqual(readable_time_difference(td), expected)


if __name__ == "__main__":
    unittest.main()
