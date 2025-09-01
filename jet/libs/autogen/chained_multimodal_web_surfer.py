import os
import shutil
import re
import time
import logging
from typing import List, Self
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
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_agentchat.utils import content_to_str, remove_images
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.agents.web_surfer._multimodal_web_surfer import UserContent, MultimodalWebSurferConfig
from autogen_ext.agents.web_surfer._events import WebSurferEvent
from autogen_ext.agents.web_surfer._prompts import (
    WEB_SURFER_QA_PROMPT,
    WEB_SURFER_QA_SYSTEM_MESSAGE,
    WEB_SURFER_TOOL_PROMPT_MM,
    WEB_SURFER_TOOL_PROMPT_TEXT,
)
from autogen_ext.agents.web_surfer._set_of_mark import add_set_of_mark
from autogen_ext.agents.web_surfer._tool_definitions import (
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
)
from playwright.async_api import BrowserContext, Download, Page, Playwright, async_playwright


class ChainedMultimodalWebSurferConfig(MultimodalWebSurferConfig):
    max_chain_steps: int = 10


class ChainedMultimodalWebSurfer(MultimodalWebSurfer):
    component_config_schema = ChainedMultimodalWebSurferConfig

    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        max_chain_steps: int = 10,
        downloads_folder: str | None = None,
        description: str = MultimodalWebSurfer.DEFAULT_DESCRIPTION,
        debug_dir: str | None = None,
        headless: bool = True,
        start_page: str | None = MultimodalWebSurfer.DEFAULT_START_PAGE,
        animate_actions: bool = False,
        to_save_screenshots: bool = False,
        use_ocr: bool = False,
        browser_channel: str | None = None,
        browser_data_dir: str | None = None,
        to_resize_viewport: bool = True,
        playwright: Playwright | None = None,
        context: BrowserContext | None = None,
    ):
        super().__init__(
            name=name,
            model_client=model_client,
            downloads_folder=downloads_folder,
            description=description,
            debug_dir=debug_dir,
            headless=headless,
            start_page=start_page,
            animate_actions=animate_actions,
            to_save_screenshots=to_save_screenshots,
            use_ocr=use_ocr,
            browser_channel=browser_channel,
            browser_data_dir=browser_data_dir,
            to_resize_viewport=to_resize_viewport,
            playwright=playwright,
            context=context,
        )
        self.max_chain_steps = max_chain_steps

    async def _generate_reply(self, cancellation_token: CancellationToken) -> UserContent:
        if not self.did_lazy_init:
            await self._lazy_init()
        assert self._page is not None
        page = self._page
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
        llm_history: List[LLMMessage] = history + [user_request]
        chain_step = 0
        while True:
            if chain_step >= self.max_chain_steps:
                return "Maximum chain steps reached without a final answer."
            rects = await self._playwright_controller.get_interactive_rects(page)
            viewport = await self._playwright_controller.get_visual_viewport(page)
            screenshot = await page.screenshot()
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
                        url=page.url,
                        message="Screenshot: " + screenshot_png_name,
                    )
                )
            tools = self.default_tools.copy()
            if viewport["pageTop"] > 5:
                tools.append(TOOL_SCROLL_UP)
            if (viewport["pageTop"] + viewport["height"] + 5) < viewport["scrollHeight"]:
                tools.append(TOOL_SCROLL_DOWN)
            focused = await self._playwright_controller.get_focused_rect_id(page)
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
            page_title = await page.title()
            prompt_message = None
            if self._model_client.model_info["vision"]:
                text_prompt = WEB_SURFER_TOOL_PROMPT_MM.format(
                    state_description=state_description,
                    visible_targets=visible_targets,
                    other_targets_str=other_targets_str,
                    focused_hint=focused_hint,
                    tool_names=tool_names,
                    title=page_title,
                    url=page.url,
                ).strip()
                scaled_screenshot = som_screenshot.resize(
                    (self.MLM_WIDTH, self.MLM_HEIGHT))
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
                    url=page.url,
                ).strip()
                prompt_message = UserMessage(content=re.sub(
                    r"(\n\s*){3,}", "\n\n", text_prompt), source=self.name)
            content_chunks = []
            final_result = None
            async for chunk in self._model_client.create_stream(
                llm_history + [prompt_message],
                tools=tools,
                extra_create_args={"tool_choice": "auto",
                                   "parallel_tool_calls": False},
                cancellation_token=cancellation_token,
            ):
                if isinstance(chunk, str):
                    content_chunks.append(chunk)
                else:
                    final_result = chunk
                    self.model_usage.append(chunk.usage)
            if final_result is None:
                return "No final result received from stream."
            message = final_result.content
            self._last_download = None
            if isinstance(message, str):
                return message
            elif isinstance(message, list):
                llm_history.append(AssistantMessage(
                    content=message, source=self.name))
                obs = await self._execute_tool(message, rects, tool_names, cancellation_token=cancellation_token)
                llm_history.append(UserMessage(content=obs, source="tool"))
            else:
                return f"Unknown response format '{message}'"
            chain_step += 1

    def _to_config(self) -> ChainedMultimodalWebSurferConfig:
        config = super()._to_config()
        return ChainedMultimodalWebSurferConfig(**config.dict(), max_chain_steps=self.max_chain_steps)

    @classmethod
    def _from_config(cls, config: ChainedMultimodalWebSurferConfig) -> Self:
        surfer = super()._from_config(config)
        surfer.max_chain_steps = config.max_chain_steps
        return surfer
