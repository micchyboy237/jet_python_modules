import io
import os
import shutil
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

OUTPUT_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "generated")
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


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
        level: Literal["DEBUG", "INFO", "WARNING",
                       "ERROR", "CRITICAL"] = "DEBUG",
        fmt: Union[str, logging.Formatter] = "%(message)s",
    ):
        self.log_file = log_file
        if self.log_file:
            log_dir = os.path.dirname(os.path.abspath(self.log_file))
            os.makedirs(log_dir, exist_ok=True)
        self.name = name
        self.overwrite = overwrite
        self.console_level = console_level.upper()
        self.level = level.upper()
        formatter = fmt if isinstance(
            fmt, logging.Formatter) else logging.Formatter(fmt)
        self.formatter = formatter
        self.logger = self._initialize_logger(name)
        self._last_message_flushed = False
        print(
            f"DEBUG: Initialized logger with console_level: {self.console_level}\nlog_file: {self.log_file}")

    def _initialize_logger(self, name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()

        # Use sys.stdout for console output
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)

        if self.log_file:
            if self.overwrite and os.path.exists(self.log_file):
                os.remove(self.log_file)
            file_handler = logging.FileHandler(self.log_file, mode='a')
            file_handler.setLevel(self.level)
            file_handler.setFormatter(self.formatter)
            logger.addHandler(file_handler)
        return logger

    def addHandler(self, handler: logging.Handler) -> None:
        self.logger.addHandler(handler)

    def removeHandler(self, handler: logging.Handler) -> None:
        self.logger.removeHandler(handler)

    def set_level(self, level: Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]) -> None:
        if isinstance(level, int):
            level_name = logging.getLevelName(level)
            if isinstance(level_name, str):
                level = level_name
            else:
                level = "DEBUG"
        elif isinstance(level, str):
            level = level.upper()
        else:
            raise TypeError("Level must be an int or a string")

        for handler in self.logger.handlers:
            handler.setLevel(level)
        print(f"DEBUG: Set logger level to {level}")

    def setLevel(self, level: Union[int, Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]]) -> None:
        self.set_level(level)

    def set_format(self, fmt: Union[str, logging.Formatter]) -> None:
        formatter = fmt if isinstance(
            fmt, logging.Formatter) else logging.Formatter(fmt)
        for handler in self.logger.handlers:
            handler.setFormatter(formatter)

    def setFormat(self, fmt: Union[str, logging.Formatter]) -> None:
        self.set_format(fmt)

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
        if force:
            self.logger.handlers.clear()

        formatter = logging.Formatter(
            fmt=format,
            datefmt=datefmt,
            style=style
        )
        self.formatter = formatter

        if filename:
            self.log_file = filename
            self.overwrite = filemode == "w"
            if self.overwrite and os.path.exists(self.log_file):
                os.remove(self.log_file)
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
                self.level if level is None else level.upper())
            file_handler.setFormatter(self.formatter)
            self.logger.addHandler(file_handler)

        if stream is not None:
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)
            console_handler = logging.StreamHandler(stream)
            console_handler.setLevel(
                self.console_level if level is None else level.upper())
            console_handler.setFormatter(self.formatter)
            self.logger.addHandler(console_handler)

        if handlers is not None:
            self.logger.handlers.clear()
            for handler in handlers:
                handler.setFormatter(self.formatter)
                if level is not None:
                    handler.setLevel(level.upper())
                self.logger.addHandler(handler)

        if level is not None:
            self.console_level = level.upper()
            self.level = level.upper()
            for handler in self.logger.handlers:
                handler.setLevel(self.console_level)

        print(
            f"DEBUG: Configured logger with filename={filename}, level={level}, format={format}")

    def basicConfig(
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
        overwrite: bool = False,
    ) -> None:
        # If overwrite is True, set filemode to "w" (write/truncate)
        if overwrite:
            filemode = "w"
        # Ensure the directory for the log file exists
        if filename is not None:
            log_dir = os.path.dirname(os.path.abspath(filename))
            os.makedirs(log_dir, exist_ok=True)
        self.set_config(
            filename=filename,
            filemode=filemode,
            format=format,
            datefmt=datefmt,
            style=style,
            level=level,
            stream=stream,
            handlers=handlers,
            force=force,
            encoding=encoding,
            errors=errors,
        )

    def custom_logger_method(self, level: str) -> Callable[[str, Optional[bool]], None]:
        def wrapper(
            message: Any,
            *args: Any,
            bright: bool = False,
            flush: bool = False,
            end: str = None,
            colors: list[str] = None,
            exc_info: bool = True,
            log_file: Optional[str] = None,
        ) -> None:
            level_map = {
                "DEBUG": 10,
                "INFO": 20,
                "WARNING": 30,
                "ERROR": 40,
                "CRITICAL": 50
            }
            if level_map.get(level.upper(), 10) < level_map.get(self.console_level, 10):
                return
            message = str(message)
            if args:
                try:
                    message = message % args
                    formatted_args = ""
                except (TypeError, ValueError) as e:
                    formatted_args = tuple(map(str, args))
            else:
                formatted_args = ""
            if colors is None:
                colors = [f"BRIGHT_{level}" if bright else level]
            else:
                colors = [f"BRIGHT_{c}" if bright and c.upper(
                ) in level_map else c for c in colors]
                colors = colors * \
                    ((len(formatted_args) + 1 + len(colors) - 1) // len(colors))
            messages = [message] + [str(arg) for arg in formatted_args]
            processed_messages = []
            for i, msg in enumerate(messages):
                parsed_message = parse_json(msg)
                if isinstance(parsed_message, (dict, list)):
                    msg = format_json(parsed_message)
                msg = fix_and_unidecode(msg) if isinstance(
                    msg, str) else str(msg)
                processed_messages.append(
                    (msg, colors[i % len(colors)] if colors else level))
            colored_output = ""
            try:
                if hasattr(sys.stdout, 'fileno') and os.isatty(sys.stdout.fileno()):
                    colored_output = "".join(
                        f"{COLORS.get(color, COLORS['LOG'])}{msg}{RESET}" for msg, color in processed_messages)
                else:
                    colored_output = " ".join(
                        msg for msg, _ in processed_messages)
            except io.UnsupportedOperation:
                colored_output = " ".join(msg for msg, _ in processed_messages)
                print(
                    f"[WARNING] Fallback to non-colored output due to io.UnsupportedOperation")
            if level.lower() == "error" and exc_info:
                error_msg = colorize_log("Trace exception:", "gray")
                try:
                    if not (hasattr(sys.stdout, 'fileno') and os.isatty(sys.stdout.fileno())):
                        error_msg = clean_ansi(error_msg)
                except io.UnsupportedOperation:
                    error_msg = clean_ansi(error_msg)
                    print(
                        f"[WARNING] Fallback to non-colored error message due to io.UnsupportedOperation")
                print(error_msg, flush=True)
                error_trace = colorize_log(traceback.format_exc(), level)
                try:
                    if not (hasattr(sys.stdout, 'fileno') and os.isatty(sys.stdout.fileno())):
                        error_trace = clean_ansi(error_trace)
                except io.UnsupportedOperation:
                    error_trace = clean_ansi(error_trace)
                    print(
                        f"[WARNING] Fallback to non-colored error trace due to io.UnsupportedOperation")
                print(error_trace, flush=True)
            if not end:
                end = "" if flush else "\n"
            print(colored_output, end=end, flush=True)
            target_log_file = log_file if log_file is not None else self.log_file
            if target_log_file:
                log_dir = os.path.dirname(os.path.abspath(target_log_file))
                os.makedirs(log_dir, exist_ok=True)
                if log_file is not None and self.overwrite and os.path.exists(target_log_file):
                    os.remove(target_log_file)
                end = "" if flush else "\n\n"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                try:
                    stack = traceback.extract_stack()
                    caller = None
                    logger_methods = {'wrapper',
                                      '__getattr__', 'custom_logger_method'}
                    for frame in reversed(stack):
                        if frame.name in logger_methods and frame.filename == __file__:
                            continue
                        caller = frame
                        break
                    if caller:
                        file_name = os.path.basename(caller.filename)
                        func_name = caller.name if caller.name != '<module>' else 'main'
                        line_number = caller.lineno
                        metadata = f"[{level.upper()}] {timestamp} {file_name}:{func_name}:{line_number}"
                    else:
                        metadata = f"[{level.upper()}] {timestamp} unknown:unknown:0"
                except IndexError:
                    metadata = f"[{level.upper()}] {timestamp} unknown:unknown:0"
                with open(target_log_file, "a") as file:
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

    def pretty(self, prompt: Any, level: int = 0, log_file: Optional[str] = None) -> None:
        MAX_STRING_LENGTH = 100

        def _inner(prompt: Any, level: int) -> str:
            prompt_log = ""
            indent = " " * level
            marker_list = ["-", "+"]
            marker = marker_list[level % 2]
            line_prefix = indent if level == 0 else f"{indent}{marker} "
            KEY_COLOR = COLORS["DEBUG"]
            VALUE_COLOR = COLORS["SUCCESS"]
            LIST_ITEM_COLOR = COLORS["SUCCESS"]

            def truncate_string(s: str) -> str:
                s = str(s)
                return s if len(s) <= MAX_STRING_LENGTH else s[:MAX_STRING_LENGTH] + "..."
            if isinstance(prompt, dict):
                for key, value in prompt.items():
                    prompt_log += f"{line_prefix}{KEY_COLOR}{str(key)}{RESET}: "
                    if isinstance(value, (dict, list)):
                        prompt_log += f"\n{_inner(value, level + 1)}"
                    else:
                        truncated_value = truncate_string(value)
                        prompt_log += f"{VALUE_COLOR}{truncated_value}{RESET}\n"
            elif isinstance(prompt, list):
                for item in prompt:
                    if isinstance(item, (dict, list)):
                        prompt_log += f"\n{_inner(item, level + 1)}"
                    else:
                        truncated_item = truncate_string(item)
                        prompt_log += f"{line_prefix}{LIST_ITEM_COLOR}{truncated_item}{RESET}\n"
            else:
                truncated_prompt = truncate_string(prompt)
                prompt_log += f"{line_prefix}{LIST_ITEM_COLOR}{truncated_prompt}{RESET}\n"
            prompt_log = fix_and_unidecode(prompt_log)
            return prompt_log
        prompt_log = _inner(prompt, level)
        try:
            if hasattr(sys.stdout, 'fileno') and os.isatty(sys.stdout.fileno()):
                print(prompt_log)
            else:
                print(clean_ansi(prompt_log))
        except io.UnsupportedOperation:
            print(clean_ansi(prompt_log))
            print(
                f"[WARNING] Fallback to non-colored output in pretty method due to io.UnsupportedOperation")
        target_log_file = log_file if log_file is not None else self.log_file
        if target_log_file:
            log_dir = os.path.dirname(os.path.abspath(target_log_file))
            os.makedirs(log_dir, exist_ok=True)
            if log_file is not None and self.overwrite and os.path.exists(target_log_file):
                os.remove(target_log_file)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                stack = traceback.extract_stack()
                caller = None
                for frame in reversed(stack):
                    if frame.name == "pretty" and frame.filename == __file__:
                        continue
                    caller = frame
                    break
                if caller:
                    file_name = os.path.basename(caller.filename)
                    func_name = caller.name
                    line_number = caller.lineno
                    metadata = f"[PRETTY] {timestamp} {file_name}:{func_name}:{line_number}"
                else:
                    metadata = f"[PRETTY] {timestamp} unknown:unknown:0"
            except IndexError:
                metadata = f"[PRETTY] {timestamp} unknown:unknown:0"
            with open(target_log_file, "a") as file:
                file.write(metadata + "\n")
                file.write(format_json(prompt) + "\n\n")
        self._last_message_flushed = False

    def __getattr__(self, name: str) -> Callable[[str, Optional[bool]], None]:
        if name.upper() in COLORS:
            return self.custom_logger_method(name.upper())
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
    logger.pretty({
        "event": "User Login",
        "details": {
            "timestamp": "2025-08-30 12:00:00",
            "user_id": 123,
            "roles": ["admin", "editor"]
        }
    }, log_file=f"{OUTPUT_DIR}/custom_pretty_log.txt")
    logger.pretty({
        "event": "Append Log",
    }, log_file=f"{OUTPUT_DIR}/custom_pretty_log.txt")
    logger.info("Splitting document ID %d into chunks", 42)
    logger.info("Hello %s, your task is complete.", "Jet")
    logger.info(
        "Analysis for %s:\n"
        "  Title: %s (Length: %d chars)\n"
        "  Word Count: %d\n"
        "  Link Count: %d\n"
        "  Image Count: %d\n"
        "  Avg Word Length: %.2f chars\n"
        "  Has Title: %s\n"
        "  Content Richness: %.2f\n"
        "  Media Ratio: %.2f%%",
        "https://blog.openai.com",
        "Error",
        5,
        0,
        0,
        0,
        72.0,
        True,
        0.0,
        0.0
    )
    logger.newline()
    logger.log(123)
    logger.success(123.12)
    logger.log("Logging to default log file (if set).")
    logger.log("Logging to custom log file.",
               log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.info("Info message to custom log file.",
                log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.warning("Warning message to custom log file.",
                   log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.error("Error message to custom log file.",
                 log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.critical("Append critical message to custom log file.",
                    log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.pretty({"example": "Append pretty message to custom log file."},
                  log_file=f"{OUTPUT_DIR}/custom_log.txt")
    logger.pretty({"example": "Append pretty message"},
                  log_file=f"{OUTPUT_DIR}/custom_pretty_log.txt")
    logger.newline()
    logger.log("Append logging with multiple arguments to custom file.", "Arg1", "Arg2",
               colors=["DEBUG", "SUCCESS"], log_file=f"{OUTPUT_DIR}/custom_log.txt")
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
    level: Literal["DEBUG", "INFO",
                   "WARNING", "ERROR", "CRITICAL"] = "DEBUG",
    fmt: Union[str, logging.Formatter] = "%(message)s",
):
    if not name or isinstance(name, str) and name == logger.name:
        return logger
    return CustomLogger(log_file, name, overwrite, console_level, level, fmt)


logger = CustomLogger()

__all__ = [
    "logger",
    "getLogger",
    "CustomLogger",
]

if __name__ == "__main__":
    args = parse_arguments()

    file_path = f"{OUTPUT_DIR}/log.txt"
    logger_with_file = CustomLogger(
        log_file=file_path,
        overwrite=False,
        console_level=args.log_cli_level
    )
    logger_examples(logger_with_file)
