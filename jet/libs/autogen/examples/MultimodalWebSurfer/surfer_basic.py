import asyncio
from jet.libs.autogen.examples.MultimodalWebSurfer.config import make_surfer, logger


async def main():
    surfer = make_surfer()

    logger.info("🚀 Visiting Python.org homepage...")
    result = await surfer.run(
        task="Visit https://www.python.org and summarize the page.")
    logger.info(f"✅ Task complete\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
