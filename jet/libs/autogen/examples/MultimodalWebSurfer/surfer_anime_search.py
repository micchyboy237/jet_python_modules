# jet_python_modules/jet/libs/autogen/examples/MultimodalWebSurfer/surfer_anime_search.py
import os
import shutil
import asyncio
import re
from typing import Any, List, Dict
from jet.logger import CustomLogger
from autogen_ext.models.ollama import OllamaChatCompletionClient
from jet.libs.autogen.multimodal_web_surfer import MultimodalWebSurfer, MultimodalWebSurferConfig
from autogen_agentchat.messages import TextMessage, MultiModalMessage
from autogen_core import CancellationToken
from autogen_agentchat.base._task import TaskResult

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
        # Task list to find streaming links
        tasks = [
            # Step 1: Perform a web search for the anime title and season with streaming keywords
            {
                "tool": "web_search",
                "args": {
                    "reasoning": f"Searching for streaming links for {anime_title} {season} to find websites offering the anime.",
                    "query": f"{anime_title} {season} streaming online watch free"
                }
            },
            # Step 2: Wait for search results to load
            {
                "tool": "sleep",
                "args": {
                    "reasoning": "Waiting for search results page to fully load to ensure all links are accessible."
                }
            },
            # Step 3: Scroll down to explore more search results
            {
                "tool": "scroll_down",
                "args": {
                    "reasoning": "Scrolling down to view additional search results that may contain streaming links."
                }
            },
            # Step 4: Wait after scrolling
            {
                "tool": "sleep",
                "args": {
                    "reasoning": "Waiting after scrolling to ensure new content is loaded."
                }
            },
            # Step 5: Summarize the page to identify potential streaming sites
            {
                "tool": "summarize_page",
                "args": {
                    "reasoning": f"Summarizing the search results page to identify websites that may offer streaming for {anime_title} {season}."
                }
            },
            # Step 6: Ask if the page contains streaming links
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Checking if the current page contains links to streaming sites for {anime_title} {season}.",
                    "question": f"Does this page contain links to websites where I can stream {anime_title} {season}?"
                }
            },
            # Step 7: Click on a potential streaming link (assumes target_id 1 is a relevant link)
            {
                "tool": "click",
                "args": {
                    "reasoning": f"Clicking on a search result link that may lead to a streaming site for {anime_title} {season}.",
                    "target_id": 1
                }
            },
            # Step 8: Wait for the streaming site to load
            {
                "tool": "sleep",
                "args": {
                    "reasoning": "Waiting for the streaming site to fully load to check for playable content."
                }
            },
            # Step 9: Verify if the page has a video player
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Verifying if the current page has a video player for streaming {anime_title} {season}.",
                    "question": f"Does this page contain a video player for streaming {anime_title} {season}?"
                }
            },
            # Step 10: Extract the current URL if it contains a video player
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Extracting the URL of the page if it contains a video player for {anime_title} {season}.",
                    "question": f"What is the URL of the current page?"
                }
            },
            # Step 11: Go back to search results
            {
                "tool": "history_back",
                "args": {
                    "reasoning": "Returning to search results to check for additional streaming links."
                }
            },
            # Step 12: Wait after navigating back
            {
                "tool": "sleep",
                "args": {
                    "reasoning": "Waiting for search results page to reload after navigating back."
                }
            },
            # Step 13: Click on another potential streaming link (assumes target_id 2)
            {
                "tool": "click",
                "args": {
                    "reasoning": f"Clicking on another search result link to find additional streaming sites for {anime_title} {season}.",
                    "target_id": 2
                }
            },
            # Step 14: Wait for the second streaming site to load
            {
                "tool": "sleep",
                "args": {
                    "reasoning": "Waiting for the second streaming site to fully load."
                }
            },
            # Step 15: Verify if the second page has a video player
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Verifying if the second page has a video player for streaming {anime_title} {season}.",
                    "question": f"Does this page contain a video player for streaming {anime_title} {season}?"
                }
            },
            # Step 16: Extract the URL of the second page if it contains a video player
            {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Extracting the URL of the second page if it contains a video player for {anime_title} {season}.",
                    "question": f"What is the URL of the current page?"
                }
            }
        ]

        # Execute tasks
        for task in tasks:
            tool_name = task["tool"]
            args = task["args"]
            message = TextMessage(
                content=f"Execute {tool_name} with args: {args}",
                source="user"
            )
            response = await agent.on_messages([message], CancellationToken())

            # Process responses for answer_question and summarize_page
            if tool_name == "answer_question":
                question = args.get("question", "")
                if isinstance(response.chat_message, (TextMessage, MultiModalMessage)):
                    content = response.chat_message.content
                    if isinstance(content, list):
                        # Extract text from multimodal content
                        content = content[0]
                    if "Does this page contain a video player" in question and "yes" in content.lower():
                        # Get the URL from the next task or previous response
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
