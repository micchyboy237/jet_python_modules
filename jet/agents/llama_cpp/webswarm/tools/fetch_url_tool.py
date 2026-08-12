import httpx
import tiktoken
from config import AtomConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _jina_fetch(url: str, api_key: str) -> str:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"https://r.jina.ai/{url}",
            headers={"Authorization": f"Bearer {api_key}", "Accept": "text/markdown"},
        )
        resp.raise_for_status()
    return resp.text


def _truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4o-mini") -> str:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + "\n\n[TRUNCATED]"


def create_fetch_url_tool(config: AtomConfig):
    llm = ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        temperature=0.0,
        max_tokens=2048,
    )

    @tool
    async def fetch_url(url: str, goal: str = "") -> str:
        """Fetch webpage content via Jina Reader and return a goal-conditioned summary."""
        raw = await _jina_fetch(url, config.jina_api_key)
        truncated = _truncate_to_tokens(raw, config.max_page_tokens, config.llm_model)

        if not goal:
            return truncated

        response = await llm.ainvoke(
            [
                {
                    "role": "system",
                    "content": "Summarize the following webpage content focusing ONLY on information relevant to the stated goal. Be concise and factual. If no relevant info exists, say 'NO_RELEVANT_INFO'.",
                },
                {"role": "user", "content": f"Goal: {goal}\n\nContent:\n{truncated}"},
            ]
        )
        return response.content.strip()

    return fetch_url
