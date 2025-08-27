from jet.libs.autogen.examples.MultimodalWebSurfer.config import make_surfer, logger


async def main():
    surfer = make_surfer()

    logger.info("❓ Asking a question about a webpage...")
    result = await surfer.run(
        task="Go to https://news.ycombinator.com and answer: 'What are the top 3 stories right now?'"
    )
    logger.info(f"✅ Answer retrieved\n{result}")
