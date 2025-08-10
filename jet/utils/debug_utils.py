from typing import Dict, Any

def get_non_function_locals() -> Dict[str, Any]:
    """
    Retrieves all local variables that are not functions from the caller's local scope.
    
    Returns:
        Dict[str, Any]: A dictionary containing non-function local variables.
    """
    local_vars = locals()
    return {name: value for name, value in local_vars.items() if not callable(value)}