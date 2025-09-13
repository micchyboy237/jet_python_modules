from typing import Generator, List, Optional
from jet.llm.ollama.base import Ollama
from jet.llm.models import OLLAMA_EMBED_MODELS, OLLAMA_MODEL_EMBEDDING_TOKENS, OLLAMA_MODEL_NAMES
from jet._token.token_utils import get_ollama_tokenizer
from jet.transformers.object import make_serializable
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.schema import BaseNode, Document, TextNode
from pydantic import BaseModel, Field
from jet.llm.ollama.constants import (
    OLLAMA_SMALL_EMBED_MODEL
)
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from jet.vectors.metadata import parse_nodes
from jet.transformers.formatters import format_json
from jet.utils.class_utils import class_to_string
from llama_index.core import SimpleDirectoryReader, PromptTemplate


class QueryData(BaseModel):
    data: List[str] = Field(
        ..., description="A list of prompts derived from query.")


DEFAULT_SAMPLE = """
Example:
```text
Provide your
```
Response:
```json
{
  "data": [
    "Provide your primary skills."
    "Provide your recent achievements.",
  ]
}
```
<end>
""".strip()


PROMPT_QA_TEMPLATE = """\
Context information is below.
---------------------
{context_str}
---------------------

Given the context information and not prior knowledge, answer the query given the partial prompt.

Query: {query_str}

Return only a single generated JSON value without any explanations surrounded by ```json that adheres to the model below:
```json
{schema_str}
```

Example prompt and response:
{sample_str}

```text
{prompt_str}
```
Response:
"""

GENERAL_QUERY = "Generate autocompletion prompts that starts with the provided partial text. Each generated prompt should be based from context."


class AutocompletePromptsGenerator:
    def __init__(self, path_or_docs: str | list[Document], model: OLLAMA_MODEL_NAMES = "llama3.2", embed_model: OLLAMA_EMBED_MODELS = OLLAMA_SMALL_EMBED_MODEL, chunk_size: Optional[int] = None, chunk_overlap: int = 100):
        final_chunk_size: int = chunk_size if isinstance(
            chunk_size, int) else OLLAMA_MODEL_EMBEDDING_TOKENS[embed_model]

        self.chunk_size = final_chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = Ollama(model=model)
        self.tokenizer = get_ollama_tokenizer(embed_model).encode
        self.documents = self._load_documents(path_or_docs)
        self.nodes = self._split_base_nodes()

    def _load_documents(self, path_or_docs: str | list[Document],  extensions: Optional[list[str]] = None) -> list[Document]:
        documents: list[Document]
        if type(path_or_docs) == str:
            documents = SimpleDirectoryReader(
                path_or_docs, required_exts=extensions, recursive=True).load_data()
        elif isinstance(path_or_docs, list):
            documents = path_or_docs
        else:
            raise ValueError(
                f"'data_dir' must be of type str | list[Document]")
        return documents

    def _split_base_nodes(self) -> list[BaseNode]:
        splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            tokenizer=self.tokenizer
        )
        all_nodes = splitter.get_nodes_from_documents(
            self.documents, show_progress=True)
        return all_nodes

    def generate(self, prompt: str, context: str) -> QueryData:
        qa_prompt_tmpl = PromptTemplate(PROMPT_QA_TEMPLATE)

        schema = QueryData.model_json_schema()
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"

        response = self.llm.structured_predict(
            QueryData,
            qa_prompt_tmpl,
            context_str=context,
            prompt_str=prompt,
            schema_str=schema,
            sample_str=DEFAULT_SAMPLE,
            query_str=GENERAL_QUERY,
            llm_kwargs={
                "options": {"temperature": 0},
            },
        )
        return response

    def process(self, prompt: str | list[str]) -> Generator[tuple[str, list[str]], None, None]:
        if not self.nodes:
            raise ValueError(f"'self.nodes' is empty")

        if isinstance(prompt, str):
            prompt = [prompt]

        context_text = "\n\n".join([node.text for node in self.nodes])

        for text in prompt:
            response = self.generate(text, context_text)
            results = response.data
            yield text, results
