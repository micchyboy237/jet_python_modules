"""Shared argument parsing utilities for Playwright-MCP examples."""

import argparse
import os
from pathlib import Path
from typing import Callable, Optional
from dotenv import load_dotenv


# ── Project root detection (robust, works from any subdirectory) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # utils/ → project2-playwright-mcp/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mcp-config.yaml"


def parse_common_args(
    description: str,
    add_extra_args_callback: Optional[Callable] = None,
) -> argparse.Namespace:
    """
    Create and parse common command-line arguments for Playwright-MCP clients.

    Args:
        description: Description text shown in --help
        add_extra_args_callback: Optional function that receives parser
                                 and can add script-specific arguments

    Returns:
        Parsed arguments namespace
    """
    # Load .env from project root if exists
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        default=os.getenv("MCP_CONFIG_PATH", str(DEFAULT_CONFIG_PATH)),
        help="Path to MCP configuration file",
    )

    parser.add_argument(
        "--url",
        "-u",
        default=os.getenv("START_URL", "https://news.ycombinator.com"),
        help="Starting URL for the browser",
    )

    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("BROWSER_HEADLESS", "false").lower() in ("true", "1", "yes", "on", "t"),
        help="Run browser in headless mode",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("DEFAULT_TIMEOUT_MS", "45000")),
        help="Default timeout in milliseconds for actions",
    )

    # Allow scripts to add their own arguments
    if add_extra_args_callback is not None:
        add_extra_args_callback(parser)

    return parser.parse_args()