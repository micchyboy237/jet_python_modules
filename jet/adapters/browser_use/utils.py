from pathlib import Path
from browser_use import Agent


def get_screenshot_paths(agent: Agent) -> list[str]:
    """Retrieve all screenshot paths from the agent's history."""
    return [
        item.state.screenshot_path
        for item in agent.history.history
        if item.state.screenshot_path is not None
    ]


def list_screenshots_in_service(agent: Agent) -> list[str]:
    """List all screenshots in the agent's screenshot service directory."""
    screenshot_dir = agent.agent_directory / "screenshots"
    return [str(p) for p in screenshot_dir.glob("*.png") if p.is_file()]
