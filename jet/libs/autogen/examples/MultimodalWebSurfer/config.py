import logging
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

# Setup logger
logger = logging.getLogger("surfer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# Ollama client (swap model if needed, e.g. "llava", "qwen2-vl")
ollama_client = OllamaChatCompletionClient(model="llama3.2")


def make_surfer(debug_dir: str = "debug_screens") -> MultimodalWebSurfer:
    return MultimodalWebSurfer(
        model_client=ollama_client,
        name="WebSurfer",
        headless=False,
        debug_dir=debug_dir,   # saves screenshots & SOM highlights here
        to_save_screenshots=True,
        start_page="http://jethros-macbook-air.local:3000"
    )
