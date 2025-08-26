import logging

DEFAULT_LOGGER = ""

# ANSI color codes
BOLD = "\u001b[1m"
RESET = "\u001b[0m"
COLORS = {
    "INFO": BOLD + "\u001b[38;5;213m",
    "BRIGHT_INFO": BOLD + "\u001b[48;5;213m",
    "DEBUG": BOLD + "\u001b[38;5;45m",
    "BRIGHT_DEBUG": BOLD + "\u001b[48;5;45m",
    "WARNING": BOLD + "\u001b[38;5;220m",
    "BRIGHT_WARNING": BOLD + "\u001b[48;5;220m",
    "ERROR": BOLD + "\u001b[38;5;196m",
    "BRIGHT_ERROR": BOLD + "\u001b[48;5;196m",
    "CRITICAL": BOLD + "\u001b[38;5;124m",
    "BRIGHT_CRITICAL": BOLD + "\u001b[48;5;124m",
    "SUCCESS": BOLD + "\u001b[38;5;40m",
    "BRIGHT_SUCCESS": BOLD + "\u001b[48;5;40m",
    "ORANGE": BOLD + "\u001b[38;5;208m",
    "BRIGHT_ORANGE": BOLD + "\u001b[48;5;208m",
    "YELLOW": BOLD + "\u001b[38;5;220m",
    "BRIGHT_YELLOW": BOLD + "\u001b[48;5;220m",
    "TEAL": BOLD + "\u001b[38;5;86m",
    "BRIGHT_TEAL": BOLD + "\u001b[48;5;86m",
    "PURPLE": BOLD + "\u001b[38;5;92m",
    "BRIGHT_PURPLE": BOLD + "\u001b[48;5;92m",
    "LIME": BOLD + "\u001b[38;5;82m",
    "BRIGHT_LIME": BOLD + "\u001b[48;5;82m",
    "CYAN": BOLD + "\u001b[38;5;51m",
    "BRIGHT_CYAN": BOLD + "\u001b[48;5;51m",
    "WHITE": BOLD + "\u001b[38;5;15m",
    "GRAY": "\u001b[38;5;250m",
    "LOG": BOLD + "\u001b[38;5;15m",
    "RESET": "\u001b[0m",
}


def colorize_log(text: str, color: str):
    color_attr = color.upper()
    if color_attr in COLORS:
        color = COLORS[color_attr]
    return f"{color}{text}{RESET}"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{
            COLORS['RESET']}"  # Colorize level name
        # Optionally colorize the message
        record.msg = f"{color}{record.msg}{COLORS['RESET']}"
        return super().format(record)


# Configure logger
def configure_logger():
    from shared.setup.logger_hooks import import_tracker, RefreshableLoggerHandler

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler()
    # formatter = ColoredFormatter("[%(levelname)s] %(message)s")
    # console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    if import_tracker:
        logger.addHandler(RefreshableLoggerHandler())

    logger.info("Configured default logging")
    return logger


# Example usage
if __name__ == "__main__":
    logger = configure_logger()

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
