"""Configuration for the summarization pipeline.

The only non-obvious piece here is `input_token_budget`: a llama.cpp server
started with `-c 10000` gives you 10k tokens total for *prompt + completion*
on a single call, not 10k tokens of input. Every call needs to leave room for
the system prompt/instructions and for the model's own output, or the request
will be truncated or rejected server-side.
"""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    model_ctx_tokens: int = 10_000          # matches llama-server's -c flag
    reserved_output_tokens: int = 700       # room for the model's completion
    system_prompt_overhead_tokens: int = 400  # room for role instructions + formatting
    temperature: float = 0.2                # low temperature: consistent, extractive summaries
    max_retries: int = 3
    retry_backoff_seconds: float = 1.5
    request_timeout_seconds: float = 120.0
    verify_sample_size: int = 3             # how many leaf facts the verifier spot-checks

    @property
    def input_token_budget(self) -> int:
        """Usable tokens for input text on a single LLM call."""
        budget = (
            self.model_ctx_tokens
            - self.reserved_output_tokens
            - self.system_prompt_overhead_tokens
        )
        if budget <= 0:
            raise ValueError(
                f"model_ctx_tokens ({self.model_ctx_tokens}) is too small once "
                f"reserved_output_tokens ({self.reserved_output_tokens}) and "
                f"system_prompt_overhead_tokens ({self.system_prompt_overhead_tokens}) "
                "are subtracted. Increase model_ctx_tokens or lower the reservations."
            )
        return budget
