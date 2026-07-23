"""md_summarizer: recursively summarize a directory of markdown files with a
small, context-limited local LLM (e.g. llama.cpp's llama-server).

Public API:
    PipelineConfig      -- token budget / model call settings
    LlamaCppClient       -- real client, talks to a running llama-server
    MockLLMClient        -- no-network client used by the demo
    run_pipeline          -- discover -> map -> reduce -> synthesize -> verify
"""

from .config import PipelineConfig
from .llm_client import LlamaCppClient, LLMRequestError, MockLLMClient
from .pipeline import run_pipeline

__all__ = [
    "PipelineConfig",
    "LlamaCppClient",
    "LLMRequestError",
    "MockLLMClient",
    "run_pipeline",
]
