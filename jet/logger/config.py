import logging

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
    "CRITICAL": BOLD + "\u001b[38;5;208m",
    "BRIGHT_CRITICAL": BOLD + "\u001b[48;5;208m",
    "SUCCESS": BOLD + "\u001b[38;5;40m",
    "BRIGHT_SUCCESS": BOLD + "\u001b[48;5;40m",
    "ORANGE": BOLD + "\u001b[38;5;208m",
    "BRIGHT_ORANGE": BOLD + "\u001b[48;5;208m",
    "PURPLE": BOLD + "\u001b[38;5;92m",
    "BRIGHT_PURPLE": BOLD + "\u001b[48;5;92m",
    "LIME": BOLD + "\u001b[38;5;82m",
    "BRIGHT_LIME": BOLD + "\u001b[48;5;82m",
    "WHITE": BOLD + "\u001b[38;5;15m",
    "GRAY": "\u001b[38;5;250m",
    "LOG": BOLD + "\u001b[38;5;15m",
    "RESET": "\u001b[0m",
}


def colorize_log(text: str, color: str):
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
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Console handler with colored formatter
    console_handler = logging.StreamHandler()
    formatter = ColoredFormatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)


# Example usage
if __name__ == "__main__":
    configure_logger()

    logging.debug("This is a debug message")
    logging.info("This is an info message")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.critical("This is a critical message")
