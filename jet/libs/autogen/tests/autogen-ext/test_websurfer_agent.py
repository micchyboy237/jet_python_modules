import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, AsyncGenerator, List, Dict
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.messages import (
    MultiModalMessage,
    TextMessage,
)
from autogen_core import CancellationToken, Image as AGImage
from autogen_core.models import AssistantMessage, UserMessage, RequestUsage
from autogen_agentchat.base import Response
from jet.libs.autogen.multimodal_web_surfer import MultimodalWebSurfer, MultimodalWebSurferConfig
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
from ollama import AsyncClient, ChatResponse, Message
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from jet.libs.autogen.playwright_controller import PlaywrightController
from pydantic import BaseModel
from PIL import Image
import io


class FileLogHandler(logging.Handler):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename
        self.file_handler = logging.FileHandler(filename)

    def emit(self, record: logging.LogRecord) -> None:
        ts = datetime.fromtimestamp(record.created).isoformat()
        if isinstance(record.msg, BaseModel):
            record.msg = json.dumps(
                {
                    "timestamp": ts,
                    "message": record.msg.model_dump_json(indent=2),
                    "type": record.msg.__class__.__name__,
                },
            )
        self.file_handler.emit(record)


logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(FileLogHandler("test_websurfer_agent.log"))


@pytest.mark.asyncio
async def test_run_websurfer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer agent with Ollama client and mocked responses
    model = "llama3.2"
    call_count = 0

    async def _mock_chat(*args: Any, **kwargs: Any) -> ChatResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ChatResponse(
                model=model,
                done=True,
                done_reason="stop",
                message=Message(
                    role="assistant",
                    content="Hello",
                ),
                prompt_eval_count=10,
                eval_count=5,
            )
        else:
            return ChatResponse(
                model=model,
                done=True,
                done_reason="tool_calls",
                message=Message(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        Message.ToolCall(
                            function=Message.ToolCall.Function(
                                name="sleep",
                                arguments={"reasoning": "sleep is important"},
                            ),
                        ),
                    ],
                ),
                prompt_eval_count=10,
                eval_count=5,
            )

    monkeypatch.setattr(AsyncClient, "chat", _mock_chat)
    # Mock add_set_of_mark to return valid PIL image and rect lists
    mock_som_image = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    mock_som_image.save(buffer, format="PNG")
    mock_som_image_bytes = buffer.getvalue()

    with patch('autogen_ext.agents.web_surfer._set_of_mark.add_set_of_mark') as mock_add_set_of_mark:
        mock_add_set_of_mark.return_value = (mock_som_image, [], [], [])
        agent = MultimodalWebSurfer(
            "WebSurfer",
            model_client=OllamaChatCompletionClient(model=model),
            use_ocr=False
        )

        # When: Running the agent with a task
        result = await agent.run(task="task")

        # Then: The agent should initialize correctly and produce expected messages
        expected_name = "WebSurfer"
        expected_message_count = 3
        expected_final_content = "Hello"
        expected_prompt_tokens = 10
        expected_completion_tokens = 5
        expected_chat_history_length = 2
        expected_chat_history_content = ["task", "Hello"]
        expected_wait_message_prefix = "I am waiting a short period of time before taking further action."

        assert agent._name == expected_name
        assert agent._playwright is not None
        assert agent._page is not None
        assert len(result.messages) == expected_message_count
        assert isinstance(result.messages[0], TextMessage)
        assert result.messages[0].models_usage is None
        assert isinstance(result.messages[1], TextMessage)
        assert isinstance(result.messages[2], TextMessage)
        assert result.messages[2].models_usage is not None
        assert result.messages[2].models_usage.completion_tokens == expected_completion_tokens
        assert result.messages[2].models_usage.prompt_tokens == expected_prompt_tokens
        assert result.messages[2].content == expected_final_content
        assert len(agent._chat_history) == expected_chat_history_length
        assert agent._chat_history[0].content == expected_chat_history_content[0]
        assert agent._chat_history[1].content == expected_chat_history_content[1]

        url_after_no_tool = agent._page.url

        # When: Running the agent again with a task that triggers a tool call
        result = await agent.run(task="task")

        # Then: The agent should handle the tool call and maintain the same URL
        assert len(result.messages) == expected_message_count
        assert isinstance(result.messages[2], MultiModalMessage)
        assert result.messages[2].content[0].startswith(
            expected_wait_message_prefix)
        url_after_sleep = agent._page.url
        assert url_after_no_tool == url_after_sleep


@pytest.mark.xfail(reason="OllamaChatCompletionClient serialization issue in autogen_ext")
@pytest.mark.asyncio
async def test_run_websurfer_declarative(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer agent with Ollama client and mocked response
    model = "llama3.2"

    async def _mock_chat(*args: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            model=model,
            done=True,
            done_reason="stop",
            message=Message(
                role="assistant",
                content="Response to message 3",
            ),
            prompt_eval_count=10,
            eval_count=5,
        )

    monkeypatch.setattr(AsyncClient, "chat", _mock_chat)
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )

    # When: Dumping and loading the agent configuration
    agent_config = agent.dump_component()
    loaded_agent = MultimodalWebSurfer.load_component(agent_config)

    # Then: The configuration and loaded agent should match expectations
    expected_provider = "autogen_ext.agents.web_surfer.MultimodalWebSurfer"
    expected_name = "WebSurfer"

    assert agent_config.provider == expected_provider
    assert agent_config.config["name"] == expected_name
    assert isinstance(loaded_agent, MultimodalWebSurfer)
    assert loaded_agent.name == expected_name


@pytest.mark.asyncio
async def test_init() -> None:
    # Given: Configuration parameters for the agent
    expected_name = "TestWebSurfer"
    expected_downloads_folder = "/tmp/downloads"
    expected_debug_dir = "/tmp/debug"
    expected_headless = True
    expected_start_page = "https://www.example.com"
    expected_animate_actions = False
    expected_to_save_screenshots = True
    expected_use_ocr = False
    expected_browser_channel = "chrome"
    expected_browser_data_dir = "/tmp/browser_data"
    expected_to_resize_viewport = True
    model_client = OllamaChatCompletionClient(model="llama3.2")

    # When: Initializing the MultimodalWebSurfer
    agent = MultimodalWebSurfer(
        name=expected_name,
        model_client=model_client,
        downloads_folder=expected_downloads_folder,
        debug_dir=expected_debug_dir,
        headless=expected_headless,
        start_page=expected_start_page,
        animate_actions=expected_animate_actions,
        to_save_screenshots=expected_to_save_screenshots,
        use_ocr=expected_use_ocr,
        browser_channel=expected_browser_channel,
        browser_data_dir=expected_browser_data_dir,
        to_resize_viewport=expected_to_resize_viewport
    )

    # Then: Attributes should be set correctly
    assert agent._name == expected_name
    assert agent.downloads_folder == expected_downloads_folder
    assert agent.debug_dir == expected_debug_dir
    assert agent.headless == expected_headless
    assert agent.start_page == expected_start_page
    assert agent.animate_actions == expected_animate_actions
    assert agent.to_save_screenshots == expected_to_save_screenshots
    assert agent.use_ocr == expected_use_ocr
    assert agent.browser_channel == expected_browser_channel
    assert agent.browser_data_dir == expected_browser_data_dir
    assert agent.to_resize_viewport == expected_to_resize_viewport
    assert agent._model_client == model_client
    assert agent._playwright is None
    assert agent._context is None
    assert agent._page is None
    assert isinstance(agent._playwright_controller, PlaywrightController)


@pytest.mark.asyncio
async def test_lazy_init(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A mocked Playwright instance and agent
    model = "llama3.2"
    expected_start_page = "https://www.bing.com/"

    async def mock_async_playwright():
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock(spec=Browser)
        mock_context = AsyncMock(spec=BrowserContext)
        mock_page = AsyncMock(spec=Page)
        mock_playwright.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page
        mock_page.set_viewport_size = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.add_init_script = AsyncMock()
        return mock_playwright

    monkeypatch.setattr(
        "playwright.async_api.async_playwright", mock_async_playwright)

    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False,
        start_page=expected_start_page
    )

    # When: Calling _lazy_init
    await agent._lazy_init()

    # Then: Browser, context, and page should be initialized
    expected_viewport = {"width": MultimodalWebSurfer.VIEWPORT_WIDTH,
                         "height": MultimodalWebSurfer.VIEWPORT_HEIGHT}
    assert agent._playwright is not None
    assert agent._context is not None
    assert agent._page is not None
    assert agent.did_lazy_init is True


@pytest.mark.asyncio
async def test_close(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: An initialized agent with mocked Playwright components
    model = "llama3.2"
    mock_playwright = AsyncMock()
    mock_context = AsyncMock(spec=BrowserContext)
    mock_page = AsyncMock(spec=Page)
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )
    agent._playwright = mock_playwright
    agent._context = mock_context
    agent._page = mock_page

    # When: Calling close
    await agent.close()

    # Then: All resources should be closed
    mock_page.close.assert_called_once()
    mock_context.close.assert_called_once()
    mock_playwright.stop.assert_called_once()
    assert agent._page is None
    assert agent._context is None
    assert agent._playwright is None


@pytest.mark.asyncio
async def test_on_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: An initialized agent with mocked PlaywrightController
    model = "llama3.2"
    mock_controller = AsyncMock()
    mock_page = AsyncMock(spec=Page)
    mock_controller.visit_page.return_value = (True, True)
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False,
        start_page="https://www.bing.com/"
    )
    agent._page = mock_page
    agent._playwright_controller = mock_controller
    agent.did_lazy_init = True
    agent._chat_history = [UserMessage(content="test", source="user")]
    agent._prior_metadata_hash = "some_hash"
    agent._last_download = Mock()

    # When: Calling on_reset
    await agent.on_reset(CancellationToken())

    # Then: State should be reset
    expected_start_page = "https://www.bing.com/"
    mock_controller.visit_page.assert_called_once_with(
        mock_page, expected_start_page)
    assert len(agent._chat_history) == 0
    assert agent._prior_metadata_hash is None
    assert agent._last_download is None


@pytest.mark.asyncio
async def test_on_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer with mocked dependencies
    model = "llama3.2"

    async def _mock_chat(*args: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            model=model,
            done=True,
            done_reason="stop",
            message=Message(
                role="assistant",
                content="Hello from web",
            ),
            prompt_eval_count=10,
            eval_count=5,
        )

    monkeypatch.setattr(AsyncClient, "chat", _mock_chat)

    # Mock add_set_of_mark to return valid PIL image and rect lists
    mock_som_image = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    mock_som_image.save(buffer, format="PNG")
    mock_som_image_bytes = buffer.getvalue()

    with patch('autogen_ext.agents.web_surfer._set_of_mark.add_set_of_mark') as mock_add_set_of_mark:
        mock_add_set_of_mark.return_value = (mock_som_image, [], [], [])
        mock_controller = AsyncMock()
        mock_page = AsyncMock(spec=Page)
        mock_controller.get_interactive_rects.return_value = {}
        mock_controller.get_visual_viewport.return_value = {
            "pageTop": 0, "height": 900, "scrollHeight": 1000}
        mock_controller.get_page_metadata.return_value = {
            "meta_tags": {"viewport": "width=device-width"}}
        mock_page.screenshot.return_value = mock_som_image_bytes
        mock_page.title.return_value = "Test Page"
        mock_page.url = "https://www.example.com"
        agent = MultimodalWebSurfer(
            "WebSurfer",
            model_client=OllamaChatCompletionClient(model=model),
            use_ocr=False
        )
        agent._page = mock_page
        agent._playwright_controller = mock_controller
        agent.did_lazy_init = True

        # When: Processing a message
        messages = [TextMessage(content="Visit example.com", source="user")]
        result = await agent.on_messages(messages, CancellationToken())

        # Then: Expected response should be returned
        expected_content = "Hello from web"
        assert isinstance(result.chat_message, TextMessage)
        assert result.chat_message.content == expected_content
        assert result.chat_message.models_usage is not None
        assert result.chat_message.models_usage.prompt_tokens == 10
        assert result.chat_message.models_usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_on_messages_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer with mocked dependencies
    model = "llama3.2"

    async def _mock_chat(*args: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            model=model,
            done=True,
            done_reason="stop",
            message=Message(
                role="assistant",
                content="Streamed response",
            ),
            prompt_eval_count=10,
            eval_count=5,
        )

    monkeypatch.setattr(AsyncClient, "chat", _mock_chat)

    # Mock add_set_of_mark to return valid PIL image and rect lists
    mock_som_image = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    mock_som_image.save(buffer, format="PNG")
    mock_som_image_bytes = buffer.getvalue()

    with patch('autogen_ext.agents.web_surfer._set_of_mark.add_set_of_mark') as mock_add_set_of_mark:
        mock_add_set_of_mark.return_value = (mock_som_image, [], [], [])
        mock_controller = AsyncMock()
        mock_page = AsyncMock(spec=Page)
        mock_controller.get_interactive_rects.return_value = {}
        mock_controller.get_visual_viewport.return_value = {
            "pageTop": 0, "height": 900, "scrollHeight": 1000}
        mock_controller.get_page_metadata.return_value = {
            "meta_tags": {"viewport": "width=device-width"}}
        mock_page.screenshot.return_value = mock_som_image_bytes
        mock_page.title.return_value = "Test Page"
        mock_page.url = "https://www.example.com"
        agent = MultimodalWebSurfer(
            "WebSurfer",
            model_client=OllamaChatCompletionClient(model=model),
            use_ocr=False
        )
        agent._page = mock_page
        agent._playwright_controller = mock_controller
        agent.did_lazy_init = True

        # When: Processing messages in streaming mode
        messages = [TextMessage(content="Stream test", source="user")]
        responses = []
        async for response in agent.on_messages_stream(messages, CancellationToken()):
            responses.append(response)

        # Then: A single response with expected content should be yielded
        expected_content = "Streamed response"
        assert len(responses) == 1
        assert isinstance(responses[0], Response)
        assert isinstance(responses[0].chat_message, TextMessage)
        assert responses[0].chat_message.content == expected_content
        assert responses[0].chat_message.models_usage is not None
        assert responses[0].chat_message.models_usage.prompt_tokens == 10
        assert responses[0].chat_message.models_usage.completion_tokens == 5


@pytest.mark.asyncio
async def test_get_state_description(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer with mocked PlaywrightController
    model = "llama3.2"
    mock_controller = AsyncMock()
    mock_page = AsyncMock(spec=Page)
    mock_controller.get_visual_viewport.return_value = {
        "pageTop": 100,
        "height": 900,
        "scrollHeight": 2000
    }
    mock_controller.get_visible_text.return_value = "Sample visible text"
    mock_page.title.return_value = "Test Page"
    mock_page.url = "https://www.example.com"
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )
    agent._page = mock_page
    agent._playwright_controller = mock_controller

    # When: Getting the state description
    result = await agent._get_state_description()

    # Then: Description should reflect viewport and page details
    expected_description = (
        "web browser is open to the page [Test Page](https://www.example.com).\n"
        "The viewport shows 45% of the webpage, and is positioned 5% down from the top of the page\n"
        "The following text is visible in the viewport:\n\nSample visible text"
    )
    assert result == expected_description


@pytest.mark.asyncio
async def test_target_name() -> None:
    # Given: A MultimodalWebSurfer and a sample rects dictionary
    model = "llama3.2"
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )
    rects = {
        "id1": {"aria_name": "Click Me", "role": "button"},
        "id2": {"role": "link"}
    }

    # When: Getting target names
    result1 = agent._target_name("id1", rects)
    result2 = agent._target_name("id2", rects)
    result3 = agent._target_name("id3", rects)

    # Then: Correct aria_name or None should be returned
    expected_result1 = "Click Me"
    expected_result2 = None
    expected_result3 = None
    assert result1 == expected_result1
    assert result2 == expected_result2
    assert result3 == expected_result3


@pytest.mark.asyncio
async def test_format_target_list() -> None:
    # Given: A MultimodalWebSurfer and sample rects
    model = "llama3.2"
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )
    rects = {
        "1": {"aria_name": "Click Me", "role": "button", "tag_name": "button"},
        "2": {"aria_name": "Search", "role": "searchbox", "tag_name": "input"},
        "3": {"aria_name": "", "role": "", "tag_name": "div"}
    }
    ids = ["1", "2", "3"]

    # When: Formatting the target list
    result = agent._format_target_list(ids, rects)

    # Then: Formatted list should match expected structure (order may vary due to set)
    expected_result = [
        '{"id": 1, "name": "Click Me", "role": "button", "tools": ["click", "hover"] }',
        '{"id": 2, "name": "Search", "role": "searchbox", "tools": ["input_text"] }',
        '{"id": 3, "name": "", "role": "div", "tools": ["click", "hover"] }'
    ]
    # Sort results to ensure deterministic comparison
    assert sorted(result) == sorted(expected_result)


@pytest.mark.asyncio
async def test_summarize_page(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: A MultimodalWebSurfer with mocked dependencies
    model = "llama3.2"

    async def _mock_chat(*args: Any, **kwargs: Any) -> ChatResponse:
        return ChatResponse(
            model=model,
            done=True,
            done_reason="stop",
            message=Message(
                role="assistant",
                content="Page summary: This is a test page.",
            ),
            prompt_eval_count=10,
            eval_count=5,
        )

    monkeypatch.setattr(AsyncClient, "chat", _mock_chat)

    # Mock screenshot to return valid PNG bytes
    mock_som_image = Image.new("RGB", (100, 100), color="white")
    buffer = io.BytesIO()
    mock_som_image.save(buffer, format="PNG")
    mock_som_image_bytes = buffer.getvalue()

    mock_controller = AsyncMock()
    mock_page = AsyncMock(spec=Page)
    mock_controller.get_page_markdown.return_value = "Test markdown content"
    mock_page.screenshot.return_value = mock_som_image_bytes
    mock_page.title.return_value = "Test Page"
    agent = MultimodalWebSurfer(
        "WebSurfer",
        model_client=OllamaChatCompletionClient(model=model),
        use_ocr=False
    )
    agent._page = mock_page
    agent._playwright_controller = mock_controller

    # When: Summarizing the page without a question
    result = await agent._summarize_page(cancellation_token=CancellationToken())

    # Then: Expected summary should be returned
    expected_summary = "Page summary: This is a test page."
    assert result == expected_summary
    assert mock_controller.get_page_markdown.called
    assert mock_page.screenshot.called

    # When: Summarizing with a question
    result = await agent._summarize_page(question="What is the main content?", cancellation_token=CancellationToken())

    # Then: Expected summary should be returned
    assert result == expected_summary
    assert mock_controller.get_page_markdown.called
    assert mock_page.screenshot.called


@pytest.mark.asyncio
async def test_to_config() -> None:
    # Given: A MultimodalWebSurfer with specific configuration
    model = "llama3.2"
    expected_name = "WebSurfer"
    expected_downloads_folder = "/tmp/downloads"
    expected_debug_dir = "/tmp/debug"
    expected_headless = True
    expected_start_page = "https://www.bing.com/"
    model_client = OllamaChatCompletionClient(model=model)
    agent = MultimodalWebSurfer(
        name=expected_name,
        model_client=model_client,
        downloads_folder=expected_downloads_folder,
        debug_dir=expected_debug_dir,
        headless=expected_headless,
        start_page=expected_start_page,
        use_ocr=False
    )

    # When: Converting to config
    config = agent._to_config()

    # Then: Config should match initialized values
    assert isinstance(config, MultimodalWebSurferConfig)
    assert config.name == expected_name
    assert config.downloads_folder == expected_downloads_folder
    assert config.debug_dir == expected_debug_dir
    assert config.headless == expected_headless
    assert config.start_page == expected_start_page
    assert config.model_client == model_client.dump_component()


@pytest.fixture(autouse=True)
async def cleanup_browser():
    # Ensure browsers are closed after each test
    yield
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        await browser.close()
