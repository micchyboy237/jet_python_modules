from typing import TypedDict, Optional
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.libs.autogen.multimodal_web_surfer import MultimodalWebSurfer
from playwright.async_api import Playwright, BrowserContext

# from jet.libs.autogen.chained_multimodal_web_surfer import ChainedMultimodalWebSurfer


class SurferConfigDict(TypedDict, total=False):
    name: str
    model_name: str  # Use model_name instead of model_client
    downloads_folder: Optional[str]
    description: Optional[str]
    debug_dir: Optional[str]
    headless: bool
    start_page: Optional[str]
    animate_actions: bool
    to_save_screenshots: bool
    use_ocr: bool
    browser_channel: Optional[str]
    browser_data_dir: Optional[str]
    to_resize_viewport: bool
    playwright: Optional[Playwright]
    context: Optional[BrowserContext]


DEFAULT_CONFIG: SurferConfigDict = {
    "name": "WebSurfer",
    "model_name": "llama3.2",  # Default model_name
    "downloads_folder": None,
    "description": None,
    "debug_dir": "debug_screens",
    "headless": False,
    "start_page": "http://jethros-macbook-air.local:3000",
    "animate_actions": True,
    "to_save_screenshots": True,
    "use_ocr": False,
    "browser_channel": "chrome",
    "browser_data_dir": "browser_data_dir",
    "to_resize_viewport": False,
    "playwright": None,
    "context": None,
}


def make_surfer(
    name: str = DEFAULT_CONFIG["name"],
    model_name: str = DEFAULT_CONFIG["model_name"],
    downloads_folder: Optional[str] = DEFAULT_CONFIG["downloads_folder"],
    description: Optional[str] = DEFAULT_CONFIG["description"],
    debug_dir: Optional[str] = DEFAULT_CONFIG["debug_dir"],
    headless: bool = DEFAULT_CONFIG["headless"],
    start_page: Optional[str] = DEFAULT_CONFIG["start_page"],
    animate_actions: bool = DEFAULT_CONFIG["animate_actions"],
    to_save_screenshots: bool = DEFAULT_CONFIG["to_save_screenshots"],
    use_ocr: bool = DEFAULT_CONFIG["use_ocr"],
    browser_channel: Optional[str] = DEFAULT_CONFIG["browser_channel"],
    browser_data_dir: Optional[str] = DEFAULT_CONFIG["browser_data_dir"],
    to_resize_viewport: bool = DEFAULT_CONFIG["to_resize_viewport"],
    playwright: Optional[Playwright] = DEFAULT_CONFIG["playwright"],
    context: Optional[BrowserContext] = DEFAULT_CONFIG["context"],
) -> MultimodalWebSurfer:
    final_config = {
        "name": name,
        "downloads_folder": downloads_folder,
        "description": description,
        "debug_dir": debug_dir,
        "headless": headless,
        "start_page": start_page,
        "animate_actions": animate_actions,
        "to_save_screenshots": to_save_screenshots,
        "use_ocr": use_ocr,
        "browser_channel": browser_channel,
        "browser_data_dir": browser_data_dir,
        "to_resize_viewport": to_resize_viewport,
        "playwright": playwright,
        "context": context,
    }
    ollama_client = OllamaChatCompletionClient(
        model=model_name)
    return MultimodalWebSurfer(
        model_client=ollama_client,
        **final_config
    )
