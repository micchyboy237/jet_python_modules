import os
import logging
from typing import List, Callable, Optional, Any
from jet.utils import COLORS, RESET


class CustomLogger:
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = self._initialize_logger()

    def _initialize_logger(self) -> logging.Logger:
        logger = logging.getLogger("CustomLogger")
        logger.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

        # File handler
        if self.log_file:
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(file_handler)

        return logger

    def custom_logger_method(self, level: str) -> Callable[[str, Optional[bool]], None]:
        def wrapper(
            *messages: list[str],
            bright: bool = False,
            flush: bool = False,
            colors: list[str] = None,
        ) -> None:
            actual_level = f"BRIGHT_{level}" if bright else level

            if colors is None:
                colors = [actual_level] * len(messages)
            else:
                colors = colors * \
                    ((len(messages) + len(colors) - 1) // len(colors))

            formatted_messages = [
                f"{COLORS.get(color, COLORS['LOG'])}{message}{RESET}" for message, color in zip(messages, colors)
            ]
            output = " ".join(formatted_messages)

            end = "" if flush else "\n"
            print(output, end=end)

            # File handler
            if self.log_file:
                with open(self.log_file, "a") as file:
                    file.write(" ".join(messages) + end)

        return wrapper

    def newline(self) -> None:
        """Prints a newline character."""
        print("\n", end="")
        if self.log_file:
            with open(self.log_file, "a") as file:
                file.write("\n")

    def __getattr__(self, name: str) -> Callable[[str, Optional[bool]], None]:
        if name.upper() in COLORS:
            return self.custom_logger_method(name.upper())
        raise AttributeError(
            f"'CustomLogger' object has no attribute '{name}'")


def logger_examples(logger: CustomLogger):
    logger.log("\n==== LOGGER METHODS =====")
    logger.newline()
    logger.log("This is a default log message.")
    logger.log("This is a default log message.", bright=True)
    logger.info("This is an info message.")
    logger.info("This is a bright info message.", bright=True)
    logger.debug("This is a debug message.")
    logger.debug("This is a bright debug message.", bright=True)
    logger.warning("This is a warning message.")
    logger.warning("This is a bright warning message.", bright=True)
    logger.error("This is an error message.")
    logger.error("This is a bright error message.", bright=True)
    logger.success("This is a success message.")
    logger.success("This is a bright success message.", bright=True)
    logger.orange("This is a orange message.")
    logger.orange("This is a bright orange message.", bright=True)
    logger.purple("This is a purple message.")
    logger.purple("This is a bright purple message.", bright=True)
    logger.lime("This is a lime message.")
    logger.lime("This is a bright lime message.", bright=True)
    logger.log("Flush word 1.", flush=True)
    logger.log("Flush word 2.", flush=True)
    logger.log("Word 1", flush=False)
    logger.log("Word 2", flush=False)
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
    logger.log("====== END LOGGER METHODS ======\n")


logger = CustomLogger()

__all__ = [
    "logger",
]

if __name__ == "__main__":
    logger_examples(logger)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(file_dir, "log.txt")
    logger_with_file = CustomLogger(log_file=file_path)
    logger_examples(logger_with_file)
