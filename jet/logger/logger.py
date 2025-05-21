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
            *messages: list[str],
            bright: bool = False,
            flush: bool = False,
            end: str = None,
            colors: list[str] = None,
            exc_info: bool = True,
        ) -> None:
            messages = list(messages)

            actual_level = f"BRIGHT_{level}" if bright else level

            if colors is None:
                colors = [actual_level] * len(messages)
            else:
                colors = colors * \
                    ((len(messages) + len(colors) - 1) // len(colors))

            if len(messages) == 1:
                if is_class_instance(messages[0]):
                    messages[0] = str(messages[0])
                else:
                    parsed_message = parse_json(messages[0])
                    if isinstance(parsed_message, (dict, list)):
                        messages[0] = format_json(parsed_message)

            messages = [
                fix_and_unidecode(message)
                if isinstance(message, str) else message
                for message in messages
            ]

            formatted_messages = [
                f"{COLORS.get(color, COLORS['LOG'])}{message}{RESET}" for message, color in zip(messages, colors)
            ]
            output = " ".join(formatted_messages)

            if level.lower() == "error" and exc_info:
                print(colorize_log("Trace exception:", "gray"))
                print(colorize_log(traceback.format_exc(), level))

            if not end:
                end = "" if flush else "\n"
            print(output, end=end)

            if self.log_file:
                end = "" if flush else "\n\n"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata = f"[{level.upper()}] {timestamp}"
                message = " ".join(str(m) for m in messages)

                with open(self.log_file, "a") as file:
                    if not flush and self._last_message_flushed:
                        file.write("\n\n")
                    if not flush or (flush and not self._last_message_flushed):
                        file.write(metadata + "\n")
                    file.write(clean_ansi(message) + end)

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
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.success("This is a success message.")
    logger.info({"key": "value", "nested": {"foo": "bar"}})
    logger.pretty({
        "user": "Alice",
        "attributes": {
            "age": 30,
            "preferences": ["running", "cycling", {"nested": "value"}],
        }
    })
    logger.set_level("WARNING")
    logger.info("This will not be shown")
    logger.warning("This will be shown")
    logger.set_format("[%(asctime)s] [%(levelname)s] %(message)s")
    logger.error("Formatted error log")
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
