import unittest
from unittest.mock import patch, MagicMock
import time
from functools import wraps

# Assuming the original code is in a module named `decorators`
from jet.logger import time_it, sleep_countdown


class TestSleepCountdown(unittest.TestCase):
    @patch("time.sleep", MagicMock())
    @patch("time.time", side_effect=[0, 1, 2, 3, 4, 5])
    def test_sleep_countdown(self, mock_time):
        """
        Test that sleep_countdown handles the countdown correctly.
        """
        with patch("builtins.print") as mock_print:
            sleep_countdown(5)
            mock_print.assert_any_call("\rSleeping, 5s...", end='', flush=True)
            mock_print.assert_any_call("\rSleeping, 1s...", end='', flush=True)
            mock_print.assert_any_call("\rSleep completed.\n")


class TestTimeItDecorator(unittest.TestCase):
    @patch("time.time", side_effect=[0, 1, 2, 3])
    @patch("threading.Thread.start", MagicMock())
    @patch("threading.Thread.join", MagicMock())
    @patch("threading.Event.set", MagicMock())
    def test_time_it_decorator_logging(self, mock_time):
        """
        Test that the decorator logs the duration of a function correctly.
        """
        @time_it
        @wraps(self.test_time_it_decorator_logging)
        def sample_function(duration):
            time.sleep(duration)
        with patch("jet.logger.logger.opt") as mock_logger:
            mock_logger.return_value.info = MagicMock()
            sample_function(3)
            mock_logger.return_value.info.assert_called_with(
                f"\rsample_function took {3}s\n", raw=True
            )


if __name__ == "__main__":
    unittest.main()
