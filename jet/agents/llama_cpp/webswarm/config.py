import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class AtomConfig(BaseModel):
    llm_base_url: str = Field(default_factory=lambda: os.environ["LLM_BASE_URL"])
    llm_api_key: str = Field(default_factory=lambda: os.environ["LLM_API_KEY"])
    llm_model: str = Field(
        default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini")
    )
    serper_api_key: str = Field(default_factory=lambda: os.environ["SERPER_API_KEY"])
    jina_api_key: str = Field(default_factory=lambda: os.environ["JINA_API_KEY"])
    max_search_results: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_SEARCH_RESULTS", "5"))
    )
    max_page_tokens: int = Field(
        default_factory=lambda: int(os.environ.get("MAX_PAGE_TOKENS", "8000"))
    )
    atom_max_steps: int = Field(
        default_factory=lambda: int(os.environ.get("ATOM_MAX_STEPS", "30"))
    )

    class Config:
        frozen = True
