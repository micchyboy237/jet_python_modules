import os
import shutil
import asyncio
import re
from typing import Any, List, Dict, TypedDict
from jet.logger import CustomLogger
from jet.adapters.autogen.ollama_client import OllamaChatCompletionClient
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


class Task(TypedDict):
    tool: str
    args: Dict[str, Any]


class StreamingLink(TypedDict):
    name: str
    url: str


def create_search_tasks(anime_title: str, season: str) -> List[Task]:
    """
    Generate a list of tasks for searching anime streaming links.

    Args:
        anime_title: The title of the anime.
        season: The season number.

    Returns:
        List of tasks to execute.
    """
    search_query = f"{anime_title} {season} streaming online watch free"
    tasks: List[Task] = [
        {
            "tool": "web_search",
            "args": {
                "reasoning": f"Searching for streaming links for {anime_title} {season}.",
                "query": search_query
            }
        },
        {
            "tool": "get_links",
            "args": {
                "reasoning": f"Retrieving all hyperlinks from search results for {anime_title} {season} to find streaming links.",
                "visible_only": False
            }
        }
    ]
    return tasks


async def process_response(response: Any, anime_title: str, season: str) -> List[StreamingLink] | None:
    """
    Process agent response to extract streaming links if a video player is found or parse links from get_links.

    Args:
        response: The response from the agent.
        anime_title: The title of the anime.
        season: The season number.

    Returns:
        List of streaming link dictionaries or None if no valid links are found.
    """
    if isinstance(response.chat_message, (TextMessage, MultiModalMessage)):
        content = response.chat_message.content
        if isinstance(content, list):
            content = content[0]
        # Handle get_links response
        if content.startswith("Links on the current webpage:"):
            links = []
            for line in content.splitlines()[1:]:
                match = re.match(
                    r"- Text: '([^']+)', URL: (https?://[^\s]+)", line)
                if match:
                    links.append(
                        {"text": match.group(1), "href": match.group(2)})
            # Filter links with streaming-related keywords
            streaming_links = [
                {"name": f"{anime_title} {season} Stream", "url": link["href"]}
                for link in links
                if any(keyword in link["text"].lower() or keyword in link["href"].lower()
                       for keyword in ["watch", "stream", "video", "episode", "anime"])
            ]
            return streaming_links[:3]  # Limit to top 3 links
        # Handle video player check
        if "yes" in content.lower() and "video player" in content.lower():
            url_response = content
            match = re.search(r'https?://[^\s]+', url_response)
            if match:
                return [{"name": f"{anime_title} {season} Stream", "url": match.group(0)}]
    return None


async def search_anime_streaming_links(
    anime_title: str,
    season: str,
    start_page: str = "http://jethros-macbook-air.local:3000"
) -> List[StreamingLink]:
    """
    Search for streaming links for a given anime title and season.

    Args:
        anime_title: The title of the anime (e.g., "Solo Leveling").
        season: The season number (e.g., "Season 2").
        start_page: The starting page for the web search (default: local searx instance).

    Returns:
        List of streaming link dictionaries.
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
    streaming_links: List[StreamingLink] = []
    try:
        tasks = create_search_tasks(anime_title, season)
        potential_links: List[StreamingLink] = []
        for task in tasks:
            tool_name = task["tool"]
            args = task["args"]
            message = TextMessage(
                content=f"Execute {tool_name} with args: {args}",
                source="user"
            )
            response = await agent.on_messages([message], CancellationToken())
            if tool_name == "get_links":
                links = await process_response(response, anime_title, season)
                if links:
                    potential_links.extend(links)
            else:
                logger.info(f"Executed {tool_name}: {args}")

        # Dynamically create tasks to visit potential streaming links
        for link in potential_links[:3]:  # Limit to 3 links
            visit_task = {
                "tool": "visit_url",
                "args": {
                    "reasoning": f"Visiting potential streaming link for {anime_title} {season}: {link['url']}.",
                    "url": link["url"]
                }
            }
            video_check_task = {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Checking for video player on page for {anime_title} {season}.",
                    "question": f"Does this page contain a video player for streaming {anime_title} {season}?"
                }
            }
            url_task = {
                "tool": "answer_question",
                "args": {
                    "reasoning": f"Extracting URL of page with video player for {anime_title} {season}.",
                    "question": "What is the URL of the current page?"
                }
            }
            for task in [visit_task, video_check_task, url_task]:
                message = TextMessage(
                    content=f"Execute {task['tool']} with args: {task['args']}",
                    source="user"
                )
                response = await agent.on_messages([message], CancellationToken())
                if task["tool"] == "answer_question" and "video player" in task["args"]["question"].lower():
                    new_links = await process_response(response, anime_title, season)
                    if new_links:
                        streaming_links.extend(new_links)
    except Exception as e:
        logger.error(f"Error during search: {e}")
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
