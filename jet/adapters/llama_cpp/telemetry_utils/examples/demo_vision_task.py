import asyncio

from jet.adapters.llama_cpp import config
from jet_telemetry import initialize_telemetry, llm, tool

# Initialize with the same Phoenix endpoint
initialize_telemetry(service_name="vision-agent", endpoint=config.PHOENIX_BASE_URL)


@tool
async def process_image(image_path: str):
    """Simulates pre-processing an image before sending to the vision model"""
    print(f"[Tool] Processing image at {image_path}...")
    await asyncio.sleep(0.1)
    return "processed_image_data"


@llm(model_name=config.VISION_MODEL)
async def describe_image(image_data: str):
    """Calls the Vision Model endpoint"""
    print(
        f"[LLM] Analyzing image with {config.VISION_MODEL} at {config.VISION_BASE_HOST}..."
    )
    await asyncio.sleep(0.4)
    return "A detailed description of the image content."


async def main():
    print("--- Starting Vision Demo ---")
    img_data = await process_image("screenshot.png")
    description = await describe_image(img_data)
    print(f"Vision Result: {description}")


if __name__ == "__main__":
    asyncio.run(main())
