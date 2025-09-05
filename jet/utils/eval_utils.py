import ast
import re
from typing import List, Dict, Union


def find_bracketed_list(text: str) -> str:
    """
    Extract the first valid Python list from a string by matching outermost square brackets,
    accounting for nested brackets and ensuring the content is a valid list.

    Args:
        text: Input string, e.g., "[TOOL_CALLS] [{'name': 'calculate', ...}]"

    Returns:
        The substring containing the valid list, e.g., "[{'name': 'calculate', ...}]"

    Raises:
        ValueError: If no valid bracketed list is found.
    """
    start_idx = -1
    bracket_count = 0

    for i, char in enumerate(text):
        if char == '[':
            if start_idx == -1:  # Potential start of a list
                start_idx = i
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
            if bracket_count == 0 and start_idx != -1:
                candidate = text[start_idx: i + 1]
                try:
                    # Try parsing to ensure it's a valid Python list
                    parsed = ast.literal_eval(candidate)
                    if isinstance(parsed, list):
                        return candidate
                except (SyntaxError, ValueError):
                    # Not a valid list; reset and continue
                    start_idx = -1
                    bracket_count = 0
                start_idx = -1  # Reset to find the next potential list

    raise ValueError("No valid bracketed list found in the input string")


def find_braced_dict(text: str) -> str:
    """
    Extract the first valid dictionary from a string by matching outermost curly braces,
    accounting for nested braces.

    Args:
        text: Input string, e.g., "{'name': 'calculate', 'arguments': {...}}"

    Returns:
        The substring containing the dictionary, e.g., "{'name': 'calculate', ...}"

    Raises:
        ValueError: If no valid braced dictionary is found.
    """
    start_idx = -1
    brace_count = 0

    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return text[start_idx: i + 1]

    raise ValueError("No valid braced dictionary found in the input string")


def evaluate_expression(expression: str) -> float:
    """
    Evaluate a simple arithmetic expression (supports +, -, *, /).

    Args:
        expression: A string containing a simple arithmetic expression, e.g., "30423 + 6999"

    Returns:
        The result of the evaluated expression as a float.

    Raises:
        ValueError: If the expression is invalid or unsupported.
    """
    try:
        # Remove whitespace and validate expression
        expression = expression.replace(" ", "")
        if not expression:
            raise ValueError("Empty expression")

        # Split on operators, assuming format like "number op number"
        parts = re.split(r'(\+|-|\*|/)', expression)
        parts = [part.strip() for part in parts if part.strip()]

        if len(parts) != 3:
            raise ValueError(
                "Expression must be in the form 'number operator number'")

        left, op, right = parts
        try:
            left_num = float(left)
            right_num = float(right)
        except ValueError:
            raise ValueError("Invalid numbers in expression")

        if op == '+':
            return left_num + right_num
        elif op == '-':
            return left_num - right_num
        elif op == '*':
            return left_num * right_num
        elif op == '/':
            if right_num == 0:
                raise ValueError("Division by zero")
            return left_num / right_num
        else:
            raise ValueError(f"Unsupported operator: {op}")

    except Exception as e:
        raise ValueError(f"Error evaluating expression: {e}")


def parse_and_evaluate(text: str) -> Union[List, Dict]:
    """
    Parse a string containing a list or dictionary and return the parsed content.
    Args:
        text: Input string, e.g., "[{'data': '30423 + 6999'}]" or "{'value': '50 * 4'}"
    Returns:
        The parsed list or dictionary.
    Raises:
        ValueError: If no valid list or dictionary is found.
    """
    try:
        # Try to find a bracketed list first
        try:
            list_text = find_bracketed_list(text)
            parsed = ast.literal_eval(list_text)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass  # No valid list found, try dictionary next

        # Try to find a braced dictionary
        dict_text = find_braced_dict(text)
        parsed = ast.literal_eval(dict_text)
        if isinstance(parsed, dict):
            return parsed

        raise ValueError("No valid list or dictionary found")

    except (SyntaxError, ValueError) as e:
        raise ValueError(f"Error parsing: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error: {e}")


if __name__ == "__main__":
    text = "[TOOL_CALLS] [{'name': 'calculate', 'arguments': {'expression': '30423 + 6999'}}]"
    result = parse_and_evaluate(text)
    expected = [{'name': 'calculate', 'arguments': {
        'expression': '30423 + 6999'}}]
    print(f"Result of expression = {result}")
    assert (result == expected)
    print("PASSED!")
