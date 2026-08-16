import trafilatura
from agent.config import Config
from jet.adapters.llama_cpp.llm_utils import chat

EXTRACTOR_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_extractor",
        "description": (
            "Extract and VERIFY specific information from a webpage. "
            "Use AFTER web_search to confirm facts before including them in responses. "
            "For list items: extract title, year, studio, synopsis, and source credibility. "
            "For dates/events: extract exact date, venue, and official confirmation status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full URL of the webpage"},
                "goal": {
                    "type": "string",
                    "description": (
                        "Precise extraction goal. Examples: "
                        "'Verify anime title, release year, and studio for [X]'; "
                        "'Confirm NBA season start date and official source'"
                    ),
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

    # Use Jet's chat() for consistent observability + model routing
    # No tool_registry here — this is a single-turn extraction, not an agentic call
    result = chat(
        prompt_or_messages=[
            {
                "role": "user",
                "content": (
                    f"From this webpage content, extract ONLY information relevant to:\n"
                    f"GOAL: {goal}\n\nCONTENT:\n{cleaned}\n\n"
                    f"If irrelevant, say so explicitly."
                ),
            }
        ],
        model=Config.LLAMA_MODEL,
        temperature=0.0,
        max_tokens=1024,
        project_name="qwen-studio-extractor",
        phoenix_url=Config.PHOENIX_URL,
    )

    return result.content.strip()
