import subprocess
from jet.logger import logger


def activate_chrome():
    """
    Activates (focuses) the Google Chrome window on macOS.
    """
    script = 'tell application "Google Chrome" to activate'
    subprocess.run(["osascript", "-e", script])
    logger.log("Activated Google Chrome", colors=["GRAY", "ORANGE"])
