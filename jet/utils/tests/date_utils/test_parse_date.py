import pytest
from datetime import datetime
# Adjust import based on your module structure
from jet.utils.date_utils import parse_date


class TestParseDate:
    def test_parse_iso_date(self):
        # Given a valid ISO date string
        date_str = "2025-07-20"
        expected = datetime(2025, 7, 20)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_iso_datetime(self):
        # Given a valid ISO datetime string without microseconds
        date_str = "2025-07-20T14:30:00"
        expected = datetime(2025, 7, 20, 14, 30, 0)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_iso_datetime_with_microseconds(self):
        # Given a valid ISO datetime string with microseconds
        date_str = "2025-07-20T14:30:00.064176"
        expected = datetime(2025, 7, 20, 14, 30, 0, 64176)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_iso_datetime_with_utc(self):
        # Given a valid ISO datetime string with UTC marker
        date_str = "2025-07-20T14:30:00Z"
        expected = datetime(2025, 7, 20, 14, 30, 0)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_iso_datetime_with_timezone(self):
        # Given a valid ISO datetime string with timezone offset
        date_str = "2025-07-20T14:30:00+02:00"
        expected = datetime(2025, 7, 20, 14, 30, 0)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_us_date(self):
        # Given a valid US date string
        date_str = "07/20/2025"
        expected = datetime(2025, 7, 20)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_european_date(self):
        # Given a valid European date string
        date_str = "20/07/2025"
        expected = datetime(2025, 7, 20)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_verbal_date(self):
        # Given a valid verbal date string
        date_str = "July 20, 2025"
        expected = datetime(2025, 7, 20)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_parse_compact_date(self):
        # Given a valid compact date string
        date_str = "20250720"
        expected = datetime(2025, 7, 20)

        # When parsing the date string
        result = parse_date(date_str)

        # Then the result should match the expected datetime
        assert result == expected

    def test_invalid_date_format(self):
        # Given an invalid date string
        date_str = "invalid-date"

        # When parsing the date string
        # Then it should raise a ValueError
        with pytest.raises(ValueError, match="Cannot parse date string"):
            parse_date(date_str)

    def test_non_string_input(self):
        # Given a non-string input
        date_input = 12345

        # When parsing the input
        # Then it should raise a ValueError
        with pytest.raises(ValueError, match="Input must be a string"):
            parse_date(date_input)
