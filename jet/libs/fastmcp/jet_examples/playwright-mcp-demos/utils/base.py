# utils/base.py

import os
import shutil
import yaml
from pathlib import Path
from fastmcp import Client
from mcp.types import Tool

from jet.utils.inspect_utils import get_entry_file_dir, get_entry_file_name

# ── Project root detection (robust, works from any subdirectory) ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # utils/ → project2-playwright-mcp/
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "mcp-config.yaml"

BASE_OUTPUT_DIR = Path(get_entry_file_dir()) / "generated" / os.path.splitext(get_entry_file_name())[0]
shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_client(config: str = str(DEFAULT_CONFIG_PATH)) -> Client:
    # Load config
    with open(config, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    client = Client(config_dict)
    return client

async def get_tools(client: Client) -> list[Tool]:
    """
    Fetch all available tools and return them as a list of Tool objects.

    Note: These objects need to be converted with `model_dump` if you wish to export 
    them in a JSON-serializable format.
    """
    raw_tools = await client.list_tools()
    return raw_tools

def get_output_dir() -> str:
    base_dir = os.path.splitext(get_entry_file_name())[0]
    return str(base_dir)
