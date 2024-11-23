from langchain_ollama.llms import OllamaLLM
from langchain_core.outputs.llm_result import LLMResult, GenerationChunk
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from jet.logger import logger


class CustomStreamingHandler(StreamingStdOutCallbackHandler):
    """Custom callback handler to log each new token with `logger.success`."""

    def on_llm_new_token(self, token: str, **kwargs: dict) -> None:
        logger.success(token, end="", flush=True)


class Ollama:
    DEFAULT_SETTINGS = {
        "seed": 42,
        "temperature": 1,
        "top_k": 40,
        "top_p": 0.5,
        "tfs_z": 1.9,
        "stop": [],
        "num_keep": 1,
        "num_predict": -2,
    }

    def __init__(self, model: str = "mistral", base_url: str = "http://jetairm1:11434"):
        self.model = model
        self.base_url = base_url
        self.ollama = OllamaLLM(model=model, base_url=base_url)

    def generate(self, prompt: str, settings: dict[str, any] = None, raw: bool = False) -> dict[str, any]:
        # Merge default settings with user-provided settings
        settings = {**self.DEFAULT_SETTINGS, **(settings or {})}

        data: LLMResult = self.ollama.generate(
            prompts=[prompt],
            options={"stream": True, "raw": raw, **settings},
            callbacks=[CustomStreamingHandler()],
        )
        generated_chunk: GenerationChunk = data.generations[0][0]
        output = generated_chunk.text.strip()

        return {
            "prompt": prompt,
            "output": output,
            "meta": {
                "prompt_len": len(prompt),
                "output_len": len(output),
                "total_len": len(prompt) + len(output),
            },
            "settings": settings,
        }


# Main function to demonstrate sample usage
if __name__ == "__main__":
    prompt = "Write a creative story about an explorer finding a hidden treasure."
    generator = Ollama()
    result = generator.generate(prompt)
    print("Generated Output:")
    print(result["output"])
