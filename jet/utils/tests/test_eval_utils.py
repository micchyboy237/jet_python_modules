import pytest
from typing import Dict, List, Union
from jet.utils.eval_utils import find_bracketed_list, find_braced_dict, evaluate_expression, parse_and_evaluate


class TestParseAndEvaluate:
    def test_valid_list(self):
        # Given: A string containing a list with a dictionary
        input_text: str = "[TOOL_CALLS] [{'name': 'calculate', 'arguments': {'expression': '30423 + 6999'}}]"
        expected_result: List[Dict] = [
            {'name': 'calculate', 'arguments': {'expression': '30423 + 6999'}}]
        # When: Parsing the input text
        result: Union[List, Dict] = parse_and_evaluate(input_text)
        # Then: The result should match the expected list
        assert result == expected_result, f"Expected {expected_result}, but got {result}"

    def test_valid_dict(self):
        # Given: A string containing a dictionary
        input_text: str = "{'value': '50 * 4'}"
        expected_result: Dict = {'value': '50 * 4'}
        # When: Parsing the input text
        result: Union[List, Dict] = parse_and_evaluate(input_text)
        # Then: The result should match the expected dictionary
        assert result == expected_result, f"Expected {expected_result}, but got {result}"

    def test_no_valid_list_or_dict(self):
        # Given: A string with no valid list or dictionary
        input_text: str = "[invalid"
        expected_error: str = "No valid list or dictionary found"
        # When: Parsing the input text
        with pytest.raises(ValueError) as exc_info:
            parse_and_evaluate(input_text)
        # Then: The expected error message should be raised
        assert expected_error in str(
            exc_info.value), f"Expected error message containing '{expected_error}', but got '{exc_info.value}'"
