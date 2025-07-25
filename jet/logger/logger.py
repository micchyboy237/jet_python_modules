import os
import sys
import logging
import traceback
import unidecode
import argparse
from datetime import datetime
from typing import List, Callable, Optional, Any, Union, Literal, Iterable
from jet.logger.config import DEFAULT_LOGGER, COLORS, RESET, colorize_log
from jet.transformers.formatters import format_json
from jet.transformers.json_parsers import parse_json
from jet.utils.text import fix_and_unidecode
from jet.utils.class_utils import is_class_instance


def clean_ansi(text: str) -> str:
    import re
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)


class CustomLogger:
    def __init__(
        self,
        log_file: Optional[str] = None,
        name: str = DEFAULT_LOGGER,
        overwrite: bool = False,
        console_level: Literal["DEBUG", "INFO",
                               "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
        file_level: Literal["DEBUG", "INFO",
                            "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
        fmt: Union[str, logging.Formatter] = "%(message)s",
    ):
        self.log_file = log_file
        self.name = name
        self.overwrite = overwrite
        self.console_level = console_level.upper()
        self.file_level = file_level.upper()
        # Initialize formatter first to ensure it's available before _initialize_logger
        formatter = fmt if isinstance(
            fmt, logging.Formatter) else logging.Formatter(fmt)
        self.formatter = formatter
        # Initialize logger after formatter is set
        self.logger = self._initialize_logger(name)
        self._last_message_flushed = False
        # Debug log to inspect initialization
        print(
            f"DEBUG: Initialized logger with console_level={self.console_level}, log_file={self.log_file}")

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

    def set_level(self, level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]) -> None:
        level = level.upper()
        for handler in self.logger.handlers:
            handler.setLevel(level)
        print(f"DEBUG: Set logger level to {level}")

    def set_format(self, fmt: Union[str, logging.Formatter]) -> None:
        formatter = fmt if isinstance(
            fmt, logging.Formatter) else logging.Formatter(fmt)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def set_config(
        self,
        *,
        filename: Optional[str] = None,
        filemode: str = "a",
        format: str = "%(message)s",
        datefmt: Optional[str] = None,
        style: Literal["%", "{", "$"] = "%",
        level: Optional[Literal["DEBUG", "INFO",
                                "WARNING", "ERROR", "CRITICAL"]] = None,
        stream: Optional[Any] = None,
        handlers: Optional[Iterable[logging.Handler]] = None,
        force: bool = False,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
    ) -> None:
        """
        Configure the logger with settings similar to logging.basicConfig.

        Args:
            filename: If specified, log messages to this file.
            filemode: Mode to open the file ('a' for append, 'w' for overwrite). Default is 'a'.
            format: Format string for log messages. Default is '%(message)s'.
            datefmt: Format string for timestamps in log messages.
            style: Format style for the formatter ('%', '{', or '$'). Default is '%'.
            level: Logging level for all handlers. If None, existing levels are unchanged.
            stream: Stream for console output. If specified, replaces existing StreamHandler.
            handlers: Iterable of handlers to set. If specified, replaces all existing handlers.
            force: If True, clear all existing handlers before applying new configuration.
            encoding: Encoding for the file handler, if filename is specified.
            errors: Error handling scheme for the file handler, if filename is specified.
        """
        # Clear existing handlers if force is True
        if force:
            self.logger.handlers.clear()

        # Update formatter with the provided format, datefmt, and style
        formatter = logging.Formatter(
            fmt=format,
            datefmt=datefmt,
            style=style
        )
        self.formatter = formatter

        # Update log file and file-related settings
        if filename:
            self.log_file = filename
            self.overwrite = filemode == "w"
            if self.overwrite and os.path.exists(self.log_file):
                os.remove(self.log_file)
            # Remove existing FileHandler, if any
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            file_handler = logging.FileHandler(
                filename,
                mode=filemode,
                encoding=encoding,
                errors=errors
            )
            file_handler.setLevel(
                self.file_level if level is None else level.upper())
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        # Update console handler with stream, if provided
        if stream is not None:
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            console_handler = logging.StreamHandler(stream)
            console_handler.setLevel(
                self.console_level if level is None else level.upper())
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        # Replace all handlers if handlers are provided
        if handlers is not None:
            self.logger.handlers.clear()
            for handler in handlers:
                handler.setFormatter(self.formatter)
                if level is not None:
                    handler.setLevel(level.upper())
                self.logger.addHandler(handler)

        # Update levels for all handlers if level is provided
        if level is not None:
            self.console_level = level.upper()
            self.file_level = level.upper()
            for handler in self.logger.handlers:
                handler.setLevel(self.console_level)

        # Debug log to inspect configuration
        print(
            f"DEBUG: Configured logger with filename={filename}, level={level}, format={format}")

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
            # Map string levels to numeric values for comparison
            level_map = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50
            }
            if level_map.get(level.upper(), 10) < level_map.get(self.console_level, 10):
                return  # Skip logging if level is below console_level

            # Handle % formatting if message contains format specifiers and args are provided
            if "%" in message and args:
                try:
                    message = message % args
                    args = ()  # Clear args after formatting to avoid duplicate processing
                except (TypeError, ValueError) as e:
                    # If formatting fails, log the error and proceed with original message
                    self.warning(
                        f"Failed to format message '{message}' with args {args}: {str(e)}")

            # Prepare colors list
            if colors is None:
                colors = [f"BRIGHT_{level}" if bright else level]
            else:
                colors = [f"BRIGHT_{c}" if bright and c.upper(
                ) in level_map else c for c in colors]
                # Extend colors list to match number of arguments
                colors = colors * \
                    ((len(args) + 1 + len(colors) - 1) // len(colors))

            # Process message and arguments
            messages = [message] + list(map(str, args))
            processed_messages = []

            for i, msg in enumerate(messages):
                if is_class_instance(msg):
                    msg = str(msg)
                else:
                    parsed_message = parse_json(msg)
                    if isinstance(parsed_message, (dict, list)):
                        msg = format_json(parsed_message)

                msg = fix_and_unidecode(msg) if isinstance(
                    msg, str) else str(msg)
                processed_messages.append(
                    (msg, colors[i % len(colors)] if colors else level))

            # Build colored output
            colored_output = ""
            if os.isatty(sys.stdout.fileno()):
                colored_output = "".join(
                    f"{COLORS.get(color, COLORS['LOG'])}{msg}{RESET}"
                    for msg, color in processed_messages
                )
            else:
                colored_output = " ".join(msg for msg, _ in processed_messages)
            final_output = colored_output

            if level.lower() == "error" and exc_info:
                error_msg = colorize_log("Trace exception:", "gray")
                if not os.isatty(sys.stdout.fileno()):
                    error_msg = clean_ansi(error_msg)
                print(error_msg)
                error_trace = colorize_log(traceback.format_exc(), level)
                if not os.isatty(sys.stdout.fileno()):
                    error_trace = clean_ansi(error_trace)
                print(error_trace)

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
                    file.write(clean_ansi(
                        " ".join(msg for msg, _ in processed_messages)) + end)

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
        # Apply colors only if stdout is a terminal
        if os.isatty(sys.stdout.fileno()):
            print(prompt_log)
        else:
            print(clean_ansi(prompt_log))

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
        # raise AttributeError(
        #     f"'CustomLogger' object has no attribute '{name}'")
        self.warning(f"'CustomLogger' object has no attribute '{name}'")


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
    logger.critical("This is a critical message.")
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
    logger.log("multi-color default", "Message 2")
    logger.log("2 multi-color with colors",
               "Message 2", colors=["DEBUG", "SUCCESS"])
    logger.log("2 multi-color cycle", "Message 2",
               "Message 3", "Message 4", "Message 5", colors=["DEBUG", "SUCCESS"])
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Custom Logger Script")
    parser.add_argument(
        "--log-cli-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the console logging level (options: DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    args = parser.parse_args()
    print(f"DEBUG: Parsed arguments: log-cli-level={args.log_cli_level}")
    return args


def getLogger(
    name: str = DEFAULT_LOGGER,
    log_file: Optional[str] = None,
    overwrite: bool = False,
    console_level: Literal["DEBUG", "INFO",
                           "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    file_level: Literal["DEBUG", "INFO",
                        "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    fmt: Union[str, logging.Formatter] = "%(message)s",
):
    """
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
    """
    if not name or isinstance(name, str) and name == logger.name:
        return logger
    return CustomLogger(log_file, name, overwrite, console_level, file_level, fmt)


logger = CustomLogger()

__all__ = [
    "logger",
    "getLogger",
    "CustomLogger",
]

if __name__ == "__main__":
    args = parse_arguments()
    logger = CustomLogger(console_level=args.log_cli_level)
    logger_examples(logger)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(file_dir, "log.txt")
    logger_with_file = CustomLogger(
        log_file=file_path,
        overwrite=True,
        console_level=args.log_cli_level
    )
    logger_examples(logger_with_file)
