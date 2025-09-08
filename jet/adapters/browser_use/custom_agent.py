from pathlib import Path
from browser_use import Agent, BrowserProfile


class CustomAgent(Agent):
    def __init__(self, *args, custom_screenshot_dir: str | Path = None, **kwargs):
        super().__init__(*args, **kwargs)
        if custom_screenshot_dir:
            self.agent_directory = Path(
                custom_screenshot_dir).resolve()  # Override temp dir
        self._set_screenshot_service()  # Re-init service with new dir
