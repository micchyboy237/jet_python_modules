"""
Formatting and display utilities.
"""


def format_output(value: any) -> str:
    """Format a value for console display."""
    return f"Result: {value}"


def log_action(action_name: str, details: str = "") -> None:
    """Log an action with optional details."""
    print(f"[LOG] {action_name}: {details}")
