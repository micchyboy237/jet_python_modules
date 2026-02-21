# example_usage.py
from dotenv import load_dotenv

load_dotenv()

from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_pipeline import (
    RAGPipeline,
)

pipeline = RAGPipeline()
pipeline.ingest(
    "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/crawl4ai/docs/md_v2/advanced"
)  # or list of files
answer = pipeline.query("What are the key features in the docs?")
print(answer)
