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
}


def colorize_log(text: str, color: str):
    return f"{color}{text}{RESET}"
