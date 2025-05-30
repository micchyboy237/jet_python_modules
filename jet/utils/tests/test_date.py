import unittest
from datetime import datetime, date
from jet.logger import logger
from typing import Union
from jet.utils.date import is_date_greater, is_date_greater_or_equal, is_date_lesser, is_date_lesser_or_equal


class TestIsDateGreaterOrEqualThan(unittest.TestCase):
    def test_date_strings(self):
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27", "2024-02-26"))
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27", "2024-02-27"))
        self.assertFalse(is_date_greater_or_equal(
            "2024-02-25", "2024-02-27"))

    def test_datetime_objects(self):
        self.assertTrue(is_date_greater_or_equal(
            datetime(2024, 2, 27, 15, 30), "2024-02-27"))
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27", datetime(2024, 2, 26, 23, 59)))
        self.assertTrue(is_date_greater_or_equal(
            datetime(2024, 2, 27, 15, 30), datetime(2024, 2, 26, 10, 10)))
        self.assertTrue(is_date_greater_or_equal(
            datetime(2024, 2, 27, 15, 30), datetime(2024, 2, 27, 23, 59)))

    def test_date_objects(self):
        self.assertTrue(is_date_greater_or_equal(
            date(2024, 2, 27), date(2024, 2, 26)))
        self.assertFalse(is_date_greater_or_equal(
            date(2024, 2, 26), date(2024, 2, 27)))

    def test_invalid_date_format(self):
        with self.assertRaises(ValueError):
            is_date_greater_or_equal("invalid-date", "2024-02-27")
        with self.assertRaises(ValueError):
            is_date_greater_or_equal("2024-02-27", "invalid-date")

    def test_iso_format_variations(self):
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27T00:00:00", "2024-02-26T23:59:59"))
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27T15:30:00.123456", "2024-02-27T15:30:00"))
        self.assertFalse(is_date_greater_or_equal(
            "2024-02-25T23:59:59", "2024-02-26T00:00:00"))
        self.assertTrue(is_date_greater_or_equal(
            "2024-02-27T23:59:59", "2024-02-27T00:00:00"))


class TestDateComparisons(unittest.TestCase):
    def test_is_date_greater(self):
        self.assertTrue(is_date_greater("2024-02-27", "2024-02-26"))
        self.assertFalse(is_date_greater("2024-02-25", "2024-02-26"))
        self.assertFalse(is_date_greater("2024-02-26", "2024-02-26"))

    def test_is_date_lesser(self):
        self.assertTrue(is_date_lesser("2024-02-25", "2024-02-26"))
        self.assertFalse(is_date_lesser("2024-02-27", "2024-02-26"))

    def test_is_date_lesser_or_equal_than(self):
        self.assertTrue(is_date_lesser_or_equal(
            "2024-02-26", "2024-02-26"))
        self.assertTrue(is_date_lesser_or_equal(
            "2024-02-25", "2024-02-26"))
        self.assertFalse(is_date_lesser_or_equal(
            "2024-02-27", "2024-02-26"))


if __name__ == "__main__":
    unittest.main()
