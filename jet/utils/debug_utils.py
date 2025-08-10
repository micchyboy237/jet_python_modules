import sys
from typing import Dict, Any, Set


def get_non_function_locals(exclude_vars: Set[str] = None) -> Dict[str, Any]:
    """
    Retrieves all local variables that are not functions from the caller's local scope.

    Args:
        exclude_vars: Set of variable names to exclude from the result (e.g., test-related variables).
                      Defaults to common pytest variables.

    Returns:
        Dict[str, Any]: A dictionary containing non-function local variables.
    """
    if exclude_vars is None:
        exclude_vars = {'self', 'setup_locals', 'expected'}

    try:
        caller_locals = sys._getframe(1).f_locals
        return {
            name: value
            for name, value in caller_locals.items()
            if not callable(value) and name not in exclude_vars
        }
    except AttributeError:
        return {}
