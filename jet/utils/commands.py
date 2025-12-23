
import json
import pyperclip
from typing import Any
from jet.transformers.object import make_serializable
from jet.logger import logger

def copy_to_clipboard(text: Any) -> None:
    """
    Copy text to clipboard using pyperclip.
    Assumes pyperclip is installed (recommended and required for perfect Unicode support on Windows).
    """
    try:
        if not isinstance(text, str):
            text = make_serializable(text)
            text = json.dumps(text, indent=2, ensure_ascii=False)

        pyperclip.copy(text)
        logger.log("[bold green]Copied to clipboard[/] (via pyperclip)", len(text), "chars")
    except Exception as e:
        logger.print_exception()
        raise RuntimeError(f"Failed to copy to clipboard: {e}")



def copy_test_result(result, expected, **kwargs):
    import inspect
    func_name = inspect.currentframe().f_back.f_code.co_name

    # Format additional kwargs if provided
    kwargs_section = ""
    if kwargs:
        formatted_kwargs = []
        for key, value in kwargs.items():
            if not isinstance(value, str):
                value = make_serializable(value)
                value = json.dumps(value, indent=2, ensure_ascii=False)
            formatted_kwargs.append(f"{key}:\n{value}")
        kwargs_section = "\n\nAdditional:\n" + "\n\n".join(formatted_kwargs)

    copy_to_clipboard(
        f"{func_name}:\n\nResult:\n{result}\n\nExpected:\n{expected}{kwargs_section}"
    )


__all__ = [
    "copy_to_clipboard",
    "copy_test_result",
]
