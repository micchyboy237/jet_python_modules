import logging

DEFAULT_LOGGER = ""

# ANSI color codes
# Color table: https://codehs.com/uploads/7c2481e9158534231fcb3c9b6003d6b3
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
    # Updated to a lighter purple for iTerm2 dark theme (Color 141 is lighter than 92)
    "PURPLE": BOLD + "\u001b[38;5;141m",
    "BRIGHT_PURPLE": BOLD + "\u001b[48;5;141m",
    "LIME": BOLD + "\u001b[38;5;82m",
    "BRIGHT_LIME": BOLD + "\u001b[48;5;82m",
    "CYAN": BOLD + "\u001b[38;5;51m",
    "BRIGHT_CYAN": BOLD + "\u001b[48;5;51m",
    # Updated to more readable magenta on dark background (205 = good saturation & contrast)
    "MAGENTA": BOLD + "\u001b[38;5;205m",
    "BRIGHT_MAGENTA": BOLD + "\u001b[48;5;205m",
    # better contrast & saturation on dark bg (#5fff87)
    "GREEN": BOLD + "\u001b[38;5;84m",          
    "BRIGHT_GREEN": BOLD + "\u001b[48;5;84m",  # or try 77 (#5fd75f) if you prefer slightly darker
    # Updated to a lighter blue for iTerm2 dark theme (Color 81 is lighter than 27)
    "BLUE": BOLD + "\u001b[38;5;81m",
    "BRIGHT_BLUE": BOLD + "\u001b[48;5;81m",
    # Updated to a more readable pink (Color 225 is lighter and has higher contrast than 218)
    "PINK": BOLD + "\u001b[38;5;212m",          # better contrast hot pink for dark bg
    "BRIGHT_PINK": BOLD + "\u001b[48;5;212m",  # or try 206 if you want more saturation
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
        # Colorize level name
        record.levelname = f"{color}{record.levelname}{COLORS['RESET']}"
        # Optionally colorize the message
        record.msg = f"{color}{record.msg}{COLORS['RESET']}"
        return super().format(record)


# Configure logger
def configure_logger():
    import sys
    from shared.setup.logger_hooks import import_tracker, RefreshableLoggerHandler

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler(stream=sys.stdout)
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
