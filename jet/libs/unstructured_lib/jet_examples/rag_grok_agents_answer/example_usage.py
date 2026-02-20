# example_usage.py
from dotenv import load_dotenv

load_dotenv()

from jet.libs.unstructured_lib.jet_examples.rag_grok_agents_answer.rag_pipeline import (
    RAGPipeline,
)

pipeline = RAGPipeline()
pipeline.ingest("path/to/your/docs/folder")  # or list of files
answer = pipeline.query("What is the key finding in the report?")
print(answer)
