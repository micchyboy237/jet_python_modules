import os
import logging
import traceback
import unidecode
from datetime import datetime

from typing import List, Callable, Optional, Any, Union
from jet.logger.config import COLORS, RESET, colorize_log
from jet.transformers.formatters import format_json
from jet.transformers.json_parsers import parse_json
from jet.utils.text import fix_and_unidecode
from jet.utils.inspect_utils import log_filtered_stack_trace
from jet.utils.class_utils import is_class_instance


def clean_ansi(text: str) -> str:
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


class CustomLogger:
    def __init__(
        self,
        log_file: Optional[str] = None,
        name: str = "default",
        overwrite: bool = False,
        console_level: str = "DEBUG",
        file_level: str = "DEBUG",
        formatter: Optional[logging.Formatter] = None,
    ):
        self.log_file = log_file
        self.overwrite = overwrite
        self.console_level = console_level
        self.file_level = file_level
        self.formatter = formatter or logging.Formatter("%(message)s")
        self.logger = self._initialize_logger(name)
        self._last_message_flushed = False

    def _initialize_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)

        if self.log_file:
            if self.overwrite and os.path.exists(self.log_file):
                os.remove(self.log_file)
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.file_level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)

        return logger

    def addHandler(self, handler: logging.Handler) -> None:
        self.logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        self.logger.removeHandler(handler)

    def set_level(self, level: str) -> None:
        for handler in self.logger.handlers:
            handler.setLevel(level.upper())

    def set_format(self, fmt: Union[str, logging.Formatter]) -> None:
        formatter = fmt if isinstance(
            fmt, logging.Formatter) else logging.Formatter(fmt)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def custom_logger_method(self, level: str) -> Callable[[str, Optional[bool]], None]:
        def wrapper(
            message: str,
            *args: Any,
            bright: bool = False,
            flush: bool = False,
            end: str = None,
            colors: list[str] = None,
            exc_info: bool = True,
        ) -> None:
            if args:
                try:
                    message = message % args
                except Exception:
                    message = f"{message} {' '.join(map(str, args))}"

            actual_level = f"BRIGHT_{level}" if bright else level

            if colors is None:
                colors = [actual_level]
            else:
                colors = colors * ((1 + len(colors) - 1) // len(colors))

            if is_class_instance(message):
                message = str(message)
            else:
                parsed_message = parse_json(message)
                if isinstance(parsed_message, (dict, list)):
                    message = format_json(parsed_message)

            message = fix_and_unidecode(message) if isinstance(
                message, str) else message

            colored_output = f"{COLORS.get(colors[0], COLORS['LOG'])}{message}{RESET}"
            final_output = colored_output

            if level.lower() == "error" and exc_info:
                print(colorize_log("Trace exception:", "gray"))
                print(colorize_log(traceback.format_exc(), level))

            if not end:
                end = "" if flush else "\n"
            print(final_output, end=end)

            if self.log_file:
                end = "" if flush else "\n\n"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata = f"[{level.upper()}] {timestamp}"

                with open(self.log_file, "a") as file:
                    if not flush and self._last_message_flushed:
                        file.write("\n\n")
                    if not flush or (flush and not self._last_message_flushed):
                        file.write(metadata + "\n")
                    file.write(clean_ansi(str(message)) + end)

                self._last_message_flushed = flush

        return wrapper

    def newline(self) -> None:
        print("\n", end="")
        if self.log_file:
            with open(self.log_file, "a") as file:
                file.write("\n")
        self._last_message_flushed = False

    def pretty(self, prompt, level=0):
        MAX_STRING_LENGTH = 100

        def _inner(prompt, level):
            prompt_log = ""
            indent = " " * level
            marker_list = ["-", "+"]
            marker = marker_list[level % 2]
            line_prefix = indent if level == 0 else f"{indent}{marker} "

            KEY_COLOR = COLORS["DEBUG"]
            VALUE_COLOR = COLORS["SUCCESS"]
            LIST_ITEM_COLOR = COLORS["SUCCESS"]

            def truncate_string(s):
                return s if len(s) <= MAX_STRING_LENGTH else s[:MAX_STRING_LENGTH] + "..."

            if isinstance(prompt, dict):
                for key, value in prompt.items():
                    prompt_log += f"{line_prefix}{KEY_COLOR}{key}{RESET}: "
                    if isinstance(value, (dict, list)):
                        prompt_log += f"\n{_inner(value, level + 1)}"
                    else:
                        truncated_value = truncate_string(str(value))
                        prompt_log += f"{VALUE_COLOR}{truncated_value}{RESET}\n"
            elif isinstance(prompt, list):
                for item in prompt:
                    if isinstance(item, (dict, list)):
                        prompt_log += f"\n{_inner(item, level + 1)}"
                    else:
                        truncated_item = truncate_string(str(item))
                        prompt_log += f"{line_prefix}{LIST_ITEM_COLOR}{truncated_item}{RESET}\n"

            prompt_log = fix_and_unidecode(prompt_log)
            return prompt_log

        prompt_log = _inner(prompt, level)
        print(prompt_log)

        if self.log_file:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata = f"[PRETTY] {timestamp}"
            with open(self.log_file, "a") as file:
                file.write(metadata + "\n")
                file.write(format_json(prompt) + "\n")
        self._last_message_flushed = False

    def __getattr__(self, name: str) -> Callable[[str, Optional[bool]], None]:
        if name.upper() in COLORS:
            return self.custom_logger_method(name.upper())
        raise AttributeError(
            f"'CustomLogger' object has no attribute '{name}'")


def logger_examples(logger: CustomLogger):
    logger.log("\n==== LOGGER METHODS =====")
    logger.newline()
    logger.log("This is a default log message.")
    logger.info("This is an info message.")
    logger.info("This is a bright info message.", bright=True)
    logger.debug("This is a debug message.")
    logger.debug("This is a bright debug message.", bright=True)
    logger.warning("This is a warning message.")
    logger.warning("This is a bright warning message.", bright=True)
    logger.error("This is an error message.")
    logger.error("This is a bright error message.", bright=True)
    logger.critical("This is an critical message.")
    logger.critical("This is a bright critical message.", bright=True)
    logger.success("This is a success message.")
    logger.success("This is a bright success message.", bright=True)
    logger.orange("This is a orange message.")
    logger.orange("This is a bright orange message.", bright=True)
    logger.teal("This is a teal message.")
    logger.teal("This is a bright teal message.", bright=True)
    logger.cyan("This is a cyan message.")
    logger.cyan("This is a bright cyan message.", bright=True)
    logger.purple("This is a purple message.")
    logger.purple("This is a bright purple message.", bright=True)
    logger.lime("This is a lime message.")
    logger.lime("This is a bright lime message.", bright=True)
    logger.log("Unicode message:", "Playwright Team \u2551 \u255a",
               colors=["WHITE", "DEBUG"])
    logger.newline()
    logger.log("Flush word 1.", flush=True)
    logger.log("Flush word 2.", flush=True)
    logger.log("Word 1", flush=False)
    logger.log("Word 2", flush=False)
    logger.newline()
    logger.log("multi-color default", "Message 2",
               "Message 3", "Message 4", "Message 5")
    logger.log("2 multi-color with colors",
               "Message 2", colors=["DEBUG", "SUCCESS"])
    logger.log("2 multi-color with bright", "Message 2",
               colors=["GRAY", "BRIGHT_DEBUG"])
    logger.log("3 multi-color", "Message 2", "Message 3",
               colors=["WHITE", "BRIGHT_DEBUG", "BRIGHT_SUCCESS"])
    logger.log("3 multi-color with repeat", "Message 2", "Message 3",
               colors=["INFO", "DEBUG"])
    logger.newline()
    logger.info({
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {
                "email": "alice@example.com",
                "phone": "123-456-7890"
            }
        },
        "status": "active"
    })
    logger.newline()
    logger.pretty({
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
            "contact": {
                "email": "alice@example.com",
                "phone": "123-456-7890"
            }
        },
        "status": "active"
    })
    logger.info("Splitting document ID %d into chunks", 42)
    logger.info("Hello %s, your task is complete.", "Jet")
    logger.newline()
    logger.log("====== END LOGGER METHODS ======\n")


logger = CustomLogger()

__all__ = [
    "logger",
    "CustomLogger",
]

if __name__ == "__main__":
    logger_examples(logger)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(file_dir, "log.txt")
    logger_with_file = CustomLogger(log_file=file_path, overwrite=True)
    logger_examples(logger_with_file)
