import os

import dspy


def configure_dspy_lm(
    temperature: float = 0.7, max_tokens: int | None = None
) -> dspy.LM:
    # For any OpenAI-compatible endpoint (vLLM, LM Studio, Ollama, LocalAI, OpenRouter, Together, Groq, Fireworks, Azure with custom endpoint, etc.)
    lm = dspy.LM(
        model=f"openai/{os.getenv('LLAMA_CPP_LLM_MODEL')}",  # ← prefix with "openai/" when using a custom base
        api_base=os.getenv("LLAMA_CPP_LLM_URL"),  # ← your custom base URL here
        # api_base="https://api.your-provider.com/v1",
        api_key="sk-1234",
        model_type="chat",  # usually "chat" for modern endpoints
        temperature=temperature,
        max_tokens=max_tokens,
        cache=False,  # optional: useful during dev
        # other kwargs go to LiteLLM / the endpoint
    )

    # Then set it globally
    dspy.configure(lm=lm)

    return lm
