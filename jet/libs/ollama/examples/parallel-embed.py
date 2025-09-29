from concurrent.futures import ThreadPoolExecutor, as_completed
from ollama import embed
from jet.logger import logger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

# Configure logger format with timestamp + level + message
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.set_format("%(asctime)s [%(levelname)s] %(message)s")

def get_embedding(text: str, model: str):
    logger.info(f"Starting embedding for model={model}, text='{text}'")
    response = embed(model=model, input=text)
    logger.info(f"Finished embedding for model={model}, text='{text}'")
    return text, model, response["embeddings"]

if __name__ == "__main__":
    inputs_and_models = [
        ("Hello, world!", "nomic-embed-text"),
        ("Parallel embedding test", "embeddinggemma"),
        ("Ollama makes embeddings easy", "mxbai-embed-large"),
    ]

    results = []
    logger.info("Submitting embedding tasks in parallel...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_input = {
            executor.submit(get_embedding, text, model): (text, model)
            for text, model in inputs_and_models
        }
        for future in as_completed(future_to_input):
            text, model, embedding = future.result()
            logger.success(f"Task complete for model={model}, text='{text}'")
            results.append((text, model, embedding))

    logger.info("All tasks completed.\n")
    for text, model, embedding in results:
        logger.debug(
            f"Result from model={model}, input='{text}', embedding first 5={embedding[:5]}"
        )
        print(f"Model: {model}\nInput: {text}\nEmbedding (first 5 values): {embedding[:5]}\n")
