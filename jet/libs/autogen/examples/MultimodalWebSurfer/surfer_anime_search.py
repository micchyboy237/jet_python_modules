import os
import shutil
import asyncio
import re
from typing import Any, List, Dict
from jet.logger import CustomLogger
from jet.libs.autogen.ollama_client import OllamaChatCompletionClient
from jet.libs.autogen.multimodal_web_surfer import (
    MultimodalWebSurfer,
    MultimodalWebSurferConfig,
    MultiModalMessage,
    TextMessage
)
from autogen_core import CancellationToken

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


async def search_anime_streaming_links(
    anime_title: str, season: str, start_page: str = "http://jethros-macbook-air.local:3000"
) -> List[Dict[str, Any]]:
    """
    Search for streaming links for a given anime title and season using MultimodalWebSurfer.
    Args:
        anime_title (str): The title of the anime (e.g., "Solo Leveling").
        season (str): The season number (e.g., "Season 2").
        start_page (str): The starting page for the web search (default: local searx instance).
    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing streaming link details.
    """
    model_client = OllamaChatCompletionClient(model="llama3.2")
    agent = MultimodalWebSurfer(
        name="AnimeWebSurfer",
        model_client=model_client,
        downloads_folder=os.path.join(OUTPUT_DIR, "downloads"),
        debug_dir=os.path.join(OUTPUT_DIR, "debug"),
        headless=False,
        start_page=start_page,
        animate_actions=False,
        to_save_screenshots=True,
        use_ocr=False,
        browser_channel="chrome",
        browser_data_dir=os.path.join(OUTPUT_DIR, "browser_data"),
        to_resize_viewport=True
    )
    streaming_links: List[Dict[str, Any]] = []
    try:
        tasks = [
            {
                "tool": "web_search",
                "args": {
                    "reasoning": f"Searching for streaming links for {anime_title} {season} to find websites offering the anime.",
                    "query": f"{anime_title} {season} streaming online watch free"
                }
            },
            {
                "tool": "summarize_page",
                "args": {
                    "reasoning": f"Summarizing the search results page to identify websites that may offer streaming for {anime_title} {season}."
                }
            },
            {
                "tool": "click",
                "args": {
                    "reasoning": f"Clicking on a search result link that may lead to a streaming site for {anime_title} {season}.",
                    "target_id": 1
                }
            },
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Verifying if the current page has a video player for streaming {anime_title} {season}.",
                    "question": f"Does this page contain a video player for streaming {anime_title} {season}?"
                }
            },
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Extracting the URL of the page if it contains a video player for {anime_title} {season}.",
                    "question": f"What is the URL of the current page?"
                }
            },
            {
                "tool": "history_back",
                "args": {
                    "reasoning": "Returning to search results to check for additional streaming links."
                }
            },
            {
                "tool": "click",
                "args": {
                    "reasoning": f"Clicking on another search result link to find additional streaming sites for {anime_title} {season}.",
                    "target_id": 2
                }
            },
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Verifying if the second page has a video player for streaming {anime_title} {season}.",
                    "question": f"Does this page contain a video player for streaming {anime_title} {season}?"
                }
            },
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Extracting the URL of the second page if it contains a video player for {anime_title} {season}.",
                    "question": f"What is the URL of the current page?"
                }
            }
        ]
        for task in tasks:
            tool_name = task["tool"]
            args = task["args"]
            message = TextMessage(
                content=f"Execute {tool_name} with args: {args}",
                source="user"
            )
            response = await agent.on_messages([message], CancellationToken())
            if tool_name == "answer_question":
                question = args.get("question", "")
                if isinstance(response.chat_message, (TextMessage, MultiModalMessage)):
                    content = response.chat_message.content
                    if isinstance(content, list):
                        content = content[0]
                    if "Does this page contain a video player" in question and "yes" in content.lower():
                        next_task_index = tasks.index(task) + 1
                        if next_task_index < len(tasks) and tasks[next_task_index]["tool"] == "answer_question":
                            url_response = await agent.on_messages(
                                [TextMessage(
                                    content=f"Execute answer_question with args: {tasks[next_task_index]['args']}",
                                    source="user"
                                )],
                                CancellationToken()
                            )
                            if isinstance(url_response.chat_message, (TextMessage, MultiModalMessage)):
                                url_content = url_response.chat_message.content
                                if isinstance(url_content, list):
                                    url_content = url_content[0]
                                match = re.search(
                                    r'https?://[^\s]+', url_content)
                                if match:
                                    streaming_links.append({
                                        "name": f"{anime_title} {season} Stream",
                                        "url": match.group(0)
                                    })
            elif tool_name == "summarize_page":
                if isinstance(response.chat_message, (TextMessage, MultiModalMessage)):
                    content = response.chat_message.content
                    if isinstance(content, list):
                        content = content[0]
                    logger.info(f"Page summary: {content}")
    except Exception as e:
        logger.error(f"Error during search: {e}")
        return streaming_links
    finally:
        try:
            await agent.close()
        except Exception as e:
            logger.error(f"Error closing agent: {e}")
        logger.info("WebSurfer agent closed.")
    return streaming_links


async def main():
    """Main function to execute the anime streaming link search."""
    anime_title = "Solo Leveling"
    season = "Season 2"
    streaming_links = await search_anime_streaming_links(anime_title, season)
    logger.info(
        f"Found {len(streaming_links)} streaming links for {anime_title} {season}:")
    for link in streaming_links:
        logger.info(f" - {link['name']}: {link['url']}")

if __name__ == "__main__":
    asyncio.run(main())
