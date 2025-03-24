from typing import Generator, List, Optional
from jet.llm.ollama.base import Ollama
from jet.transformers.object import make_serializable
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import Document, TextNode
from pydantic import BaseModel, Field
from jet.llm.ollama.constants import (
    OLLAMA_LARGE_CHUNK_OVERLAP,
    OLLAMA_LARGE_CHUNK_SIZE,
    OLLAMA_LARGE_EMBED_MODEL,
    OLLAMA_LARGE_LLM_MODEL
)
from jet.logger import logger
from jet.vectors.metadata import parse_nodes
from jet.transformers.formatters import format_json
from jet.utils.class_utils import class_to_string
from llama_index.core import SimpleDirectoryReader, PromptTemplate


class QueryData(BaseModel):
    data: List[str] = Field(
        ..., description="A list of prompts derived from query.")


DEFAULT_TONE_NAME = "an independent clause with a single direct object"
DEFAULT_FORMAT = """
{
  "data": [
    "Sentence 1"
  ]
}
""".strip()
DEFAULT_SAMPLE = """
Example 1:
```text
Create an account at my website.
```
Response:
{
  "data": []
}

Example 2:
```text
Provide your best work schedule and projects.
```
Response:
{
  "data": [
    "Provide your best work schedule.",
    "Provide your best projects."
  ]
}

Example 3:
```text
She gave her friend a book and a pen.
```
Response:
{
  "data": [
    "She gave her friend a book.",
    "She gave her friend a pen."
  ]
}
""".strip()

INSTRUCTIONS_PROMPT = """

""".strip()

PROMPT_QA_TEMPLATE = """
Break down the provided text into sentences that has a single object each.
Each sentence should have a subject and predicate.
The subject should be shared.
You may return empty if the sentence has 0 or 1 object already.

Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:
{class_str}

Example prompt and response:
{sample_str}

```text
{prompt_str}
```
Response:
""".strip()


class PromptsGenerator:
    def __init__(self, data_path: Optional[str] = None, llm: Ollama | LLM = Ollama(model="llama3.1"), chunk_size: int = 512, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.nodes = self.load_and_parse_data(data_path) if data_path else []

    def load_and_parse_data(self, data_path: str):
        logger.newline()
        logger.info("Loading data...")
        docs = SimpleDirectoryReader(data_path).load_data(show_progress=True)
        logger.log("All docs:", len(docs), colors=["DEBUG", "SUCCESS"])

        large_chunk_size = self.chunk_size
        large_chunk_overlap = self.chunk_overlap
        logger.newline()
        logger.info("Parse large nodes...")
        logger.log("Large chunk size:", large_chunk_size,
                   colors=["GRAY", "INFO"])
        logger.log("Large chunk overlap:", large_chunk_overlap,
                   colors=["GRAY", "INFO"])
        large_nodes = parse_nodes(
            docs, large_chunk_size, large_chunk_overlap)
        logger.log("Large nodes count:", len(large_nodes),
                   colors=["DEBUG", "SUCCESS"])

        # medium_chunk_size = self.chunk_size // 2
        # medium_chunk_overlap = self.chunk_overlap // 2
        # logger.newline()
        # logger.info("Parse medium nodes...")
        # logger.log("Medium chunk size:", medium_chunk_size,
        #            colors=["GRAY", "INFO"])
        # logger.log("Medium chunk overlap:", medium_chunk_overlap,
        #            colors=["GRAY", "INFO"])
        # medium_nodes = parse_nodes(
        #     docs, medium_chunk_size, medium_chunk_overlap)
        # logger.log("Medium nodes count:", len(medium_nodes),
        #            colors=["DEBUG", "SUCCESS"])

        # small_chunk_size = self.chunk_size // 4
        # small_chunk_overlap = self.chunk_overlap // 4
        # logger.newline()
        # logger.info("Parse small nodes...")
        # logger.log("Small chunk size:", small_chunk_size,
        #            colors=["GRAY", "INFO"])
        # logger.log("Small chunk overlap:", small_chunk_overlap,
        #            colors=["GRAY", "INFO"])
        # small_nodes = parse_nodes(
        #     docs, small_chunk_size, small_chunk_overlap)
        # logger.log("Small nodes count:", len(small_nodes),
        #            colors=["DEBUG", "SUCCESS"])

        base_nodes = [
            # *small_nodes,
            # *medium_nodes,
            *large_nodes,
        ]

        return base_nodes

    def generate(self, prompt: str, tone_name: str = DEFAULT_TONE_NAME, format: str = DEFAULT_FORMAT, sample: str = DEFAULT_SAMPLE) -> QueryData:
        qa_prompt_tmpl = PromptTemplate(PROMPT_QA_TEMPLATE)

        response = self.llm.structured_predict(
            QueryData,
            qa_prompt_tmpl,
            prompt_str=prompt,
            format_str=format,
            sample_str=sample,
            class_str=class_to_string(QueryData),
            llm_kwargs={
                "options": {"temperature": 0},
            },
        )
        return response

    def process(self, prompt: str | list[str]) -> Generator[tuple[str, str], None, None]:
        if not self.nodes:
            if isinstance(prompt, str):
                prompt = [prompt]
            for text in prompt:
                response = self.generate(text)
                for result in response.data:
                    yield text, result
        else:
            for node in self.nodes:
                nodes_to_parse = [node]
                texts = [node.text for node in nodes_to_parse]

                logger.log("Parsed nodes:", len(texts),
                           colors=["DEBUG", "SUCCESS"])
                context_text = "\n\n".join(texts)

                response = self.generate(context_text)
                for result in response.data:
                    yield node.text, result
