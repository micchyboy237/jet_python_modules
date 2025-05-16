import asyncio
from infinity_emb import AsyncEmbeddingEngine, EngineArgs

sentences = ["Embed this is sentence via Infinity.", "Paris is in France."]
engine = AsyncEmbeddingEngine.from_args(
    EngineArgs(model_name_or_path = "BAAI/bge-large-en-v1.5", device="cpu", engine="optimum" # or engine="torch"
))

async def main(): 
    async with engine:
        embeddings, usage = await engine.embed(sentences=sentences)
asyncio.run(main())