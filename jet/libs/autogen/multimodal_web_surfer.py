import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import sys
import time
import traceback
import warnings
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Sequence,
)
from urllib.parse import quote_plus
import aiofiles
import PIL.Image
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, MultiModalMessage, TextMessage
from autogen_agentchat.utils import content_to_str, remove_images
from autogen_core import EVENT_LOGGER_NAME, CancellationToken, Component, ComponentModel, FunctionCall
from autogen_core import Image as AGImage
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    ModelFamily,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from PIL import Image
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright
from pydantic import BaseModel
from typing_extensions import Self
from autogen_ext.agents.web_surfer._events import WebSurferEvent
from autogen_ext.agents.web_surfer._prompts import (
    WEB_SURFER_QA_PROMPT,
    WEB_SURFER_QA_SYSTEM_MESSAGE,
    WEB_SURFER_TOOL_PROMPT_MM,
    WEB_SURFER_TOOL_PROMPT_TEXT,
)
from autogen_ext.agents.web_surfer._set_of_mark import add_set_of_mark
from autogen_ext.agents.web_surfer._types import InteractiveRegion, UserContent
from jet.libs.autogen._tool_definitions import (
    TOOL_CLICK,
    TOOL_HISTORY_BACK,
    TOOL_HOVER,
    TOOL_READ_PAGE_AND_ANSWER,
    TOOL_SCROLL_DOWN,
    TOOL_SCROLL_UP,
    TOOL_SLEEP,
    TOOL_SUMMARIZE_PAGE,
    TOOL_TYPE,
    TOOL_VISIT_URL,
    TOOL_WEB_SEARCH,
    TOOL_GET_VISIBLE_LINKS,
)
from jet.libs.autogen.playwright_controller import PlaywrightController
DEFAULT_CONTEXT_SIZE = 128000


class MultimodalWebSurferConfig(BaseModel):
    name: str
    model_client: ComponentModel
    downloads_folder: str | None = None
    description: str | None = None
    debug_dir: str | None = None
    headless: bool = True
    start_page: str | None = "https://www.bing.com/"
    animate_actions: bool = False
    to_save_screenshots: bool = False
    use_ocr: bool = False
    browser_channel: str | None = None
    browser_data_dir: str | None = None
    to_resize_viewport: bool = True


class MultimodalWebSurfer(BaseChatAgent, Component[MultimodalWebSurferConfig]):
    component_type = "agent"
    component_config_schema = MultimodalWebSurferConfig
    component_provider_override = "autogen_ext.agents.web_surfer.MultimodalWebSurfer"
    DEFAULT_DESCRIPTION = """
    A helpful assistant with access to a web browser.
    Ask them to perform web searches, open pages, and interact with content (e.g., clicking links, scrolling the viewport, filling in form fields, etc.).
    It can also summarize the entire page, or answer questions based on the content of the page.
    It can also be asked to sleep and wait for pages to load, in cases where the page seems not yet fully loaded.
    """
    DEFAULT_START_PAGE = "https://www.bing.com/"
    VIEWPORT_HEIGHT = 900
    VIEWPORT_WIDTH = 1440
    MLM_HEIGHT = 765
    MLM_WIDTH = 1224
    SCREENSHOT_TOKENS = 1105

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        downloads_folder: str | None = None,
        description: str = DEFAULT_DESCRIPTION,
        debug_dir: str | None = None,
        headless: bool = True,
        start_page: str | None = DEFAULT_START_PAGE,
        animate_actions: bool = False,
        to_save_screenshots: bool = False,
        use_ocr: bool = False,
        browser_channel: str | None = None,
        browser_data_dir: str | None = None,
        to_resize_viewport: bool = True,
        playwright: Playwright | None = None,
        context: BrowserContext | None = None,
    ):
        """
        Initialize the MultimodalWebSurfer.
        """
        super().__init__(name, description)
        if debug_dir is None and to_save_screenshots:
            raise ValueError(
                "Cannot save screenshots without a debug directory. Set it using the 'debug_dir' parameter. The debug directory is created if it does not exist."
            )
        if model_client.model_info["function_calling"] is False:
            raise ValueError(
                "The model does not support function calling. MultimodalWebSurfer requires a model that supports function calling."
            )
        self._model_client = model_client
        self.headless = headless
        self.browser_channel = browser_channel
        self.browser_data_dir = browser_data_dir
        self.start_page = start_page or self.DEFAULT_START_PAGE
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        self.to_save_screenshots = to_save_screenshots
        self.use_ocr = use_ocr
        self.to_resize_viewport = to_resize_viewport
        self.animate_actions = animate_actions
        self._playwright: Playwright | None = playwright
        self._context: BrowserContext | None = context
        self._page: Page | None = None
        self._last_download: Download | None = None
        self._prior_metadata_hash: str | None = None
        self.model_usage: List[RequestUsage] = []  # Initialize model_usage
        self.logger = logging.getLogger(EVENT_LOGGER_NAME)
        self._chat_history: List[LLMMessage] = []

        def _download_handler(download: Download) -> None:
            self._last_download = download
        self._download_handler = _download_handler
        self._playwright_controller = PlaywrightController(
            animate_actions=self.animate_actions,
            downloads_folder=self.downloads_folder,
            viewport_width=self.VIEWPORT_WIDTH,
            viewport_height=self.VIEWPORT_HEIGHT,
            _download_handler=self._download_handler,
            to_resize_viewport=self.to_resize_viewport,
        )
        self.default_tools = [
            TOOL_VISIT_URL,
            TOOL_WEB_SEARCH,
            TOOL_HISTORY_BACK,
            TOOL_CLICK,
            TOOL_TYPE,
            TOOL_READ_PAGE_AND_ANSWER,
            TOOL_SUMMARIZE_PAGE,
            TOOL_SLEEP,
            TOOL_HOVER,
            TOOL_GET_VISIBLE_LINKS,
        ]
        self.did_lazy_init = False

    async def _lazy_init(
        self,
    ) -> None:
        """
        On the first call, we initialize the browser and the page.
        """
        if sys.platform == "win32":
            current_policy = asyncio.get_event_loop_policy()
            if hasattr(asyncio, "WindowsProactorEventLoopPolicy") and not isinstance(
                current_policy, asyncio.WindowsProactorEventLoopPolicy
            ):
                warnings.warn(
                    "The current event loop policy is not WindowsProactorEventLoopPolicy. "
                    "This may cause issues with subprocesses. "
                    "Try setting the event loop policy to WindowsProactorEventLoopPolicy. "
                    "For example: `asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())`. "
                    "See https://docs.python.org/3/library/asyncio-eventloop.html",
                    stacklevel=2,
                )
        self._last_download = None
        self._prior_metadata_hash = None
        launch_args: Dict[str, Any] = {"headless": self.headless}
        if self.browser_channel is not None:
            launch_args["channel"] = self.browser_channel
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._context is None:
            if self.browser_data_dir is None:
                browser = await self._playwright.chromium.launch(**launch_args)
                self._context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
                )
            else:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    self.browser_data_dir, **launch_args
                )
        self._context.set_default_timeout(60000)
        self._page = await self._context.new_page()
        assert self._page is not None
        self._page.on("download", self._download_handler)
        if self.to_resize_viewport:
            await self._page.set_viewport_size({"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(
                os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()
        await self._set_debug_dir(self.debug_dir)
        self.did_lazy_init = True

    async def close(self) -> None:
        """
        Close the browser and the page.
        Should be called when the agent is no longer needed.
        """
        if self._page is not None:
            await self._page.close()
            self._page = None
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def _set_debug_dir(self, debug_dir: str | None) -> None:
        assert self._page is not None
        if self.debug_dir is None:
            return
        if not os.path.isdir(self.debug_dir):
            os.mkdir(self.debug_dir)
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))
            self.logger.info(
                WebSurferEvent(
                    source=self.name,
                    url=self._page.url,
                    message="Screenshot: " + screenshot_png_name,
                )
            )

    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (MultiModalMessage,)

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        if not self.did_lazy_init:
            return
        assert self._page is not None
        self._chat_history.clear()
        reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
            self._page, self.start_page
        )
        if reset_last_download and self._last_download is not None:
            self._last_download = None
        if reset_prior_metadata and self._prior_metadata_hash is not None:
            self._prior_metadata_hash = None
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            await self._page.screenshot(path=os.path.join(self.debug_dir, screenshot_png_name))
            self.logger.info(
                WebSurferEvent(
                    source=self.name,
                    url=self._page.url,
                    message="Screenshot: " + screenshot_png_name,
                )
            )
        self.logger.info(
            WebSurferEvent(
                source=self.name,
                url=self._page.url,
                message="Resetting browser.",
            )
        )

    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        async for message in self.on_messages_stream(messages, cancellation_token):
            if isinstance(message, Response):
                return message
        raise AssertionError(
            "The stream should have returned the final result.")

    async def on_messages_stream(
        self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken
    ) -> AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]:
        for chat_message in messages:
            self._chat_history.append(chat_message.to_model_message())
        self.inner_messages: List[BaseAgentEvent | BaseChatMessage] = []
        self.model_usage = []  # Reset model_usage for each stream
        try:
            content = await self._generate_reply(cancellation_token=cancellation_token)
            self._chat_history.append(AssistantMessage(
                content=content_to_str(content), source=self.name))
            final_usage = RequestUsage(
                prompt_tokens=sum([u.prompt_tokens for u in self.model_usage]),
                completion_tokens=sum(
                    [u.completion_tokens for u in self.model_usage]),
            )
            if isinstance(content, str):
                yield Response(
                    chat_message=TextMessage(
                        content=content, source=self.name, models_usage=final_usage),
                    inner_messages=self.inner_messages,
                )
            else:
                yield Response(
                    chat_message=MultiModalMessage(
                        content=content, source=self.name, models_usage=final_usage),
                    inner_messages=self.inner_messages,
                )
        except BaseException:
            content = f"Web surfing error:\n\n{traceback.format_exc()}"
            self._chat_history.append(AssistantMessage(
                content=content, source=self.name))
            yield Response(chat_message=TextMessage(content=content, source=self.name))

    async def _generate_reply(self, cancellation_token: CancellationToken) -> UserContent:
        """Generates the actual reply. First calls the LLM to figure out which tool to use, then executes the tool."""
        if not self.did_lazy_init:
            await self._lazy_init()
        assert self._page is not None
        history: List[LLMMessage] = remove_images(self._chat_history)
        if len(history):
            user_request = history.pop()
        else:
            user_request = UserMessage(content="Empty request.", source="user")
        if self._model_client.model_info["family"] not in [
            ModelFamily.GPT_4O,
            ModelFamily.O1,
            ModelFamily.O3,
            ModelFamily.GPT_4,
            ModelFamily.GPT_35,
        ]:
            history = []
        rects = await self._playwright_controller.get_interactive_rects(self._page)
        viewport = await self._playwright_controller.get_visual_viewport(self._page)
        screenshot = await self._page.screenshot()
        som_screenshot, visible_rects, rects_above, rects_below = add_set_of_mark(
            screenshot, rects)
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot_som" + current_timestamp + ".png"
            som_screenshot.save(os.path.join(
                self.debug_dir, screenshot_png_name))
            self.logger.info(
                WebSurferEvent(
                    source=self.name,
                    url=self._page.url,
                    message="Screenshot: " + screenshot_png_name,
                )
            )
        tools = self.default_tools.copy()
        if viewport["pageTop"] > 5:
            tools.append(TOOL_SCROLL_UP)
        if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
            tools.append(TOOL_SCROLL_DOWN)
        focused = await self._playwright_controller.get_focused_rect_id(self._page)
        focused_hint = ""
        if focused:
            name = self._target_name(focused, rects)
            if name:
                name = f"(and name '{name}') "
            else:
                name = ""
            role = "control"
            try:
                role = rects[focused]["role"]
            except KeyError:
                pass
            focused_hint = f"\nThe {role} with ID {focused} {name}currently has the input focus.\n\n"
        visible_targets = "\n".join(
            self._format_target_list(visible_rects, rects)) + "\n\n"
        other_targets: List[str] = []
        other_targets.extend(self._format_target_list(rects_above, rects))
        other_targets.extend(self._format_target_list(rects_below, rects))
        if len(other_targets) > 0:
            if len(other_targets) > 30:
                other_targets = other_targets[0:30]
                other_targets.append("...")
            other_targets_str = (
                "Additional valid interaction targets include (but are not limited to):\n"
                + "\n".join(other_targets)
                + "\n\n"
            )
        else:
            other_targets_str = ""
        state_description = "Your " + await self._get_state_description()
        tool_names = "\n".join([t["name"] for t in tools])
        page_title = await self._page.title()
        prompt_message = None
        if self._model_client.model_info["vision"]:
            text_prompt = WEB_SURFER_TOOL_PROMPT_MM.format(
                state_description=state_description,
                visible_targets=visible_targets,
                other_targets_str=other_targets_str,
                focused_hint=focused_hint,
                tool_names=tool_names,
                title=page_title,
                url=self._page.url,
            ).strip()
            scaled_screenshot = som_screenshot.resize(
                (self.MLM_WIDTH, MLM_HEIGHT))
            som_screenshot.close()
            if self.to_save_screenshots:
                scaled_screenshot.save(os.path.join(
                    self.debug_dir, "screenshot_scaled.png"))
            prompt_message = UserMessage(
                content=[re.sub(
                    r"(\n\s*){3,}", "\n\n", text_prompt), AGImage.from_pil(scaled_screenshot)],
                source=self.name,
            )
        else:
            text_prompt = WEB_SURFER_TOOL_PROMPT_TEXT.format(
                state_description=state_description,
                visible_targets=visible_targets,
                other_targets_str=other_targets_str,
                focused_hint=focused_hint,
                tool_names=tool_names,
                title=page_title,
                url=self._page.url,
            ).strip()
            prompt_message = UserMessage(content=re.sub(
                r"(\n\s*){3,}", "\n\n", text_prompt), source=self.name)
        history.append(prompt_message)
        history.append(user_request)
        response = await self._model_client.create(
            history, tools=tools, extra_create_args={"tool_choice": "auto"}, cancellation_token=cancellation_token
        )
        self.model_usage.append(response.usage)
        message = response.content
        self._last_download = None
        if isinstance(message, str):
            self.inner_messages.append(TextMessage(
                content=message, source=self.name))
            return message
        elif isinstance(message, list):
            return await self._execute_tool(message, rects, tool_names, cancellation_token=cancellation_token)
        else:
            raise AssertionError(f"Unknown response format '{message}'")

    async def _execute_tool(
        self,
        message: List[FunctionCall],
        rects: Dict[str, InteractiveRegion],
        tool_names: str,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> UserContent:
        name = message[0].name
        args = json.loads(message[0].arguments)
        action_description = ""
        assert self._page is not None
        self.logger.info(
            WebSurferEvent(
                source=self.name,
                url=self._page.url,
                action=name,
                arguments=args,
                message=f"{name}( {json.dumps(args)} )",
            )
        )
        self.inner_messages.append(TextMessage(
            content=f"{name}( {json.dumps(args)} )", source=self.name))
        if name == "visit_url":
            url = args.get("url")
            action_description = f"I typed '{url}' into the browser address bar."
            if url.startswith(("https://", "http://", "file://", "about:")):
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, url
                )
            elif " " in url:
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, f"{self.start_page}/search?q={quote_plus(url)}"
                )
            else:
                reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                    self._page, "https://" + url
                )
            if reset_last_download and self._last_download is not None:
                self._last_download = None
            if reset_prior_metadata and self._prior_metadata_hash is not None:
                self._prior_metadata_hash = None
        elif name == "history_back":
            action_description = "I clicked the browser back button."
            await self._playwright_controller.back(self._page)
        elif name == "web_search":
            query = args.get("query")
            action_description = f"I typed '{query}' into the browser search bar."
            reset_prior_metadata, reset_last_download = await self._playwright_controller.visit_page(
                self._page, f"{self.start_page}/search?q={quote_plus(query)}"
            )
            if reset_last_download and self._last_download is not None:
                self._last_download = None
            if reset_prior_metadata and self._prior_metadata_hash is not None:
                self._prior_metadata_hash = None
        elif name == "scroll_up":
            action_description = "I scrolled up one page in the browser."
            await self._playwright_controller.page_up(self._page)
        elif name == "scroll_down":
            action_description = "I scrolled down one page in the browser."
            await self._playwright_controller.page_down(self._page)
        elif name == "click":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I clicked '{target_name}'."
            else:
                action_description = "I clicked the control."
            new_page_tentative = await self._playwright_controller.click_id(self._page, target_id)
            if new_page_tentative is not None:
                self._page = new_page_tentative
                self._prior_metadata_hash = None
                self.logger.info(
                    WebSurferEvent(
                        source=self.name,
                        url=self._page.url,
                        message="New tab or window.",
                    )
                )
        elif name == "input_text":
            input_field_id = str(args.get("input_field_id"))
            text_value = str(args.get("text_value"))
            input_field_name = self._target_name(input_field_id, rects)
            if input_field_name:
                action_description = f"I typed '{text_value}' into '{input_field_name}'."
            else:
                action_description = f"I input '{text_value}'."
            await self._playwright_controller.fill_id(self._page, input_field_id, text_value)
        elif name == "scroll_element_up":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I scrolled '{target_name}' up."
            else:
                action_description = "I scrolled the control up."
            await self._playwright_controller.scroll_id(self._page, target_id, "up")
        elif name == "scroll_element_down":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I scrolled '{target_name}' down."
            else:
                action_description = "I scrolled the control down."
            await self._playwright_controller.scroll_id(self._page, target_id, "down")
        elif name == "answer_question":
            question = str(args.get("question"))
            action_description = f"I answered the following question '{question}' based on the web page."
            return await self._summarize_page(question=question, cancellation_token=cancellation_token)
        elif name == "summarize_page":
            action_description = "I summarized the current web page"
            return await self._summarize_page(cancellation_token=cancellation_token)
        elif name == "hover":
            target_id = str(args.get("target_id"))
            target_name = self._target_name(target_id, rects)
            if target_name:
                action_description = f"I hovered over '{target_name}'."
            else:
                action_description = "I hovered over the control."
            await self._playwright_controller.hover_id(self._page, target_id)
        elif name == "sleep":
            action_description = "I am waiting a short period of time before taking further action."
            await self._playwright_controller.sleep(self._page, 3)
        elif name == "get_visible_links":
            action_description = "I retrieved the list of visible hyperlinks from the current webpage."
            links = await self._playwright_controller.get_visible_links(self._page)
            if not links:
                return "No visible links found on the current webpage."
            formatted_links = "\n".join(
                [f"- Text: '{link['text']}', URL: {link['href']}" for link in links])
            return f"Visible links on the current webpage:\n{formatted_links}"
        else:
            raise ValueError(
                f"Unknown tool '{name}'. Please choose from:\n\n{tool_names}")
        await self._page.wait_for_load_state()
        await self._playwright_controller.sleep(self._page, 3)
        if self._last_download is not None and self.downloads_folder is not None:
            fname = os.path.join(self.downloads_folder,
                                 self._last_download.suggested_filename)
            await self._last_download.save_as(fname)
            page_body = f"<html><head><title>Download Successful</title></head><body style=\"margin: 20px;\"><h1>Successfully downloaded '{self._last_download.suggested_filename}' to local path:<br><br>{fname}</h1></body></html>"
            await self._page.goto(
                "data:text/html;base64," +
                base64.b64encode(page_body.encode("utf-8")).decode("utf-8")
            )
            await self._page.wait_for_load_state()
        page_metadata = json.dumps(await self._playwright_controller.get_page_metadata(self._page), indent=4)
        metadata_hash = hashlib.md5(page_metadata.encode("utf-8")).hexdigest()
        if metadata_hash != self._prior_metadata_hash:
            page_metadata = (
                "\n\nThe following metadata was extracted from the webpage:\n\n" +
                page_metadata.strip() + "\n"
            )
        else:
            page_metadata = ""
        self._prior_metadata_hash = metadata_hash
        new_screenshot = await self._page.screenshot()
        if self.to_save_screenshots:
            current_timestamp = "_" + int(time.time()).__str__()
            screenshot_png_name = "screenshot" + current_timestamp + ".png"
            async with aiofiles.open(os.path.join(self.debug_dir, screenshot_png_name), "wb") as file:
                await file.write(new_screenshot)
            self.logger.info(
                WebSurferEvent(
                    source=self.name,
                    url=self._page.url,
                    message="Screenshot: " + screenshot_png_name,
                )
            )
        state_description = "The " + await self._get_state_description()
        message_content = (
            f"{action_description}\n\n" + state_description +
            page_metadata + "\nHere is a screenshot of the page."
        )
        return [
            re.sub(r"(\n\s*){3,}", "\n\n", message_content),
            AGImage.from_pil(PIL.Image.open(io.BytesIO(new_screenshot))),
        ]

    async def _get_state_description(self) -> str:
        assert self._playwright_controller is not None
        assert self._page is not None
        viewport = await self._playwright_controller.get_visual_viewport(self._page)
        percent_visible = int(
            viewport["height"] * 100 / viewport["scrollHeight"])
        percent_scrolled = int(
            viewport["pageTop"] * 100 / viewport["scrollHeight"])
        if percent_scrolled < 1:
            position_text = "at the top of the page"
        elif percent_scrolled + percent_visible >= 99:
            position_text = "at the bottom of the page"
        else:
            position_text = str(percent_scrolled) + \
                "% down from the top of the page"
        visible_text = await self._playwright_controller.get_visible_text(self._page)
        page_title = await self._page.title()
        message_content = f"web browser is open to the page [{page_title}]({self._page.url}).\nThe viewport shows {percent_visible}% of the webpage, and is positioned {position_text}\n"
        message_content += f"The following text is visible in the viewport:\n\n{visible_text}"
        return message_content

    def _target_name(self, target: str, rects: Dict[str, InteractiveRegion]) -> str | None:
        try:
            return rects[target]["aria_name"].strip()
        except KeyError:
            return None

    def _format_target_list(self, ids: List[str], rects: Dict[str, InteractiveRegion]) -> List[str]:
        """
        Format the list of targets in the webpage as a string to be used in the agent's prompt.
        """
        targets: List[str] = []
        for r in list(set(ids)):
            if r in rects:
                aria_role = rects[r].get("role", "").strip()
                if len(aria_role) == 0:
                    aria_role = rects[r].get("tag_name", "").strip()
                aria_name = re.sub(
                    r"[\n\r]+", " ", rects[r].get("aria_name", "")).strip()
                actions = ['"click", "hover"']
                if rects[r]["role"] in ["textbox", "searchbox", "search"]:
                    actions = ['"input_text"']
                actions_str = "[" + ",".join(actions) + "]"
                targets.append(
                    f'{{"id": {r}, "name": "{aria_name}", "role": "{aria_role}", "tools": {actions_str} }}')
        return targets

    async def _summarize_page(
        self,
        question: str | None = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        assert self._page is not None
        page_markdown: str = await self._playwright_controller.get_page_markdown(self._page)
        title: str = self._page.url
        try:
            title = await self._page.title()
        except Exception:
            pass
        screenshot = Image.open(io.BytesIO(await self._page.screenshot()))
        scaled_screenshot = screenshot.resize(
            (self.MLM_WIDTH, self.MLM_HEIGHT))
        screenshot.close()
        ag_image = AGImage.from_pil(scaled_screenshot)
        messages: List[LLMMessage] = []
        messages.append(SystemMessage(content=WEB_SURFER_QA_SYSTEM_MESSAGE))
        prompt = WEB_SURFER_QA_PROMPT(title, question)
        buffer = ""
        for line in page_markdown.splitlines():
            trial_message = UserMessage(
                content=prompt + buffer + line,
                source=self.name,
            )
            try:
                remaining = self._model_client.remaining_tokens(
                    messages + [trial_message])
            except KeyError:
                remaining = DEFAULT_CONTEXT_SIZE - \
                    self._model_client.count_tokens(messages + [trial_message])
            if self._model_client.model_info["vision"] and remaining <= 0:
                break
            if self._model_client.model_info["vision"] and remaining <= self.SCREENSHOT_TOKENS:
                break
            buffer += line
        buffer = buffer.strip()
        if len(buffer) == 0:
            return "Nothing to summarize."
        if self._model_client.model_info["vision"]:
            messages.append(
                UserMessage(
                    content=[
                        prompt + buffer,
                        ag_image,
                    ],
                    source=self.name,
                )
            )
        else:
            messages.append(
                UserMessage(
                    content=prompt + buffer,
                    source=self.name,
                )
            )
        response = await self._model_client.create(messages, cancellation_token=cancellation_token)
        self.model_usage.append(response.usage)
        scaled_screenshot.close()
        assert isinstance(response.content, str)
        return response.content

    def _to_config(self) -> MultimodalWebSurferConfig:
        return MultimodalWebSurferConfig(
            name=self.name,
            model_client=self._model_client.dump_component(),
            downloads_folder=self.downloads_folder,
            description=self.description,
            debug_dir=self.debug_dir,
            headless=self.headless,
            start_page=self.start_page,
            animate_actions=self.animate_actions,
            to_save_screenshots=self.to_save_screenshots,
            use_ocr=self.use_ocr,
            browser_channel=self.browser_channel,
            browser_data_dir=self.browser_data_dir,
            to_resize_viewport=self.to_resize_viewport,
        )

    @classmethod
    def _from_config(cls, config: MultimodalWebSurferConfig) -> Self:
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(
                config.model_client),
            downloads_folder=config.downloads_folder,
            description=config.description or cls.DEFAULT_DESCRIPTION,
            debug_dir=config.debug_dir,
            headless=config.headless,
            start_page=config.start_page or cls.DEFAULT_START_PAGE,
            animate_actions=config.animate_actions,
            to_save_screenshots=config.to_save_screenshots,
            use_ocr=config.use_ocr,
            browser_channel=config.browser_channel,
            browser_data_dir=config.browser_data_dir,
            to_resize_viewport=config.to_resize_viewport,
        )
