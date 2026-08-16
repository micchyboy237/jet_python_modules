import trafilatura
from agent.config import Config
from agent.llm_client import LLMClient

EXTRACTOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_extractor",
        "description": "Extract specific information from a webpage. Use AFTER finding a URL via web_search.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the webpage"},
                "goal": {
                    "type": "string",
                    "description": "Precise description of what to extract",
                },
            },
            "required": ["url", "goal"],
        },
    },
}


def web_extractor(url: str, goal: str) -> str:
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return "Error: Could not fetch URL."

    cleaned = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=True,
        no_fallback=True,
        favor_precision=True,
    )

    if not cleaned or len(cleaned.strip()) < 50:
        return "Error: No meaningful text content extracted."

    cleaned = cleaned[: Config.EXTRACTOR_MAX_CHARS]

    llm = LLMClient()
    msg = llm.chat(
        messages=[
            {
                "role": "user",
                "content": (
                    f"From this webpage content, extract ONLY information relevant to:\n"
                    f"GOAL: {goal}\n\nCONTENT:\n{cleaned}\n\n"
                    f"If irrelevant, say so explicitly."
                ),
            }
        ]
    )
    return msg.content.strip()
