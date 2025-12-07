
from jet.logger import logger
import pyperclip

def copy_to_clipboard(text: str) -> None:
    """
    Copy text to clipboard using pyperclip.
    Assumes pyperclip is installed (recommended and required for perfect Unicode support on Windows).
    """
    try:
        pyperclip.copy(text)
        logger.log("[bold green]Copied to clipboard[/] (via pyperclip)", len(text), "chars")
    except Exception as e:
        logger.print_exception()
        raise RuntimeError(f"Failed to copy to clipboard: {e}")



def copy_test_result(result, expected):
    import inspect
    func_name = inspect.currentframe().f_back.f_code.co_name
    copy_to_clipboard(
        f"{func_name}:\n\nResult:\n{result}\n\nExpected:\n{expected}")


__all__ = [
    "copy_to_clipboard",
    "copy_test_result",
]
