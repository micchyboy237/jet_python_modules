import asyncio
from jet.libs.autogen.examples.MultimodalWebSurfer.config import make_surfer, logger


async def main():
    surfer = make_surfer()

    logger.info("🔍 Starting multi-step navigation...")
    result = await surfer.run(
        task="Search for 'fastapi site:fastapi.tiangolo.com', click the first result, then scroll down and summarize."
    )
    logger.info(f"✅ Navigation complete\n{result}")

if __name__ == "__main__":
    asyncio.run(main())
