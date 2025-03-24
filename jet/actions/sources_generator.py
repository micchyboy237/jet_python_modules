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
from jet.llm.ollama.base import initialize_ollama_settings
from jet.logger import logger
from jet.vectors.metadata import parse_nodes
from jet.transformers.formatters import format_json
from jet.utils.class_utils import class_to_string
from llama_index.core import SimpleDirectoryReader, PromptTemplate


class Source(BaseModel):
    filename: str = Field(
        ...,
        description="The name of the file where the source is found.")
    lines: List[int] = Field(
        ...,
        description="The start and end line numbers containing the source text.")
    question: str = Field(
        ...,
        description="Short question text relevant to the source text from context information provided.")
    answer: str = Field(
        ...,
        description="Answer to the question.")


class QueryData(BaseModel):
    data: List[Source] = Field(
        ..., description="A list of sources related to the query.")


# SOURCE_QA_TEMPLATE = """\
# Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. Every answer should include at least one source source. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
# Query: When is water wet?
# Answer: {
#   "data": [
#     {
#         "filename": "filename_1.txt",
#         "lines": [1, 3],
#         "question": "Question 1"
#         "answer": "Answer 1"
#     }
#   ]
# }
# Now it's your turn. Below are context information containing several numbered sources of information:
# ------
# {context_str}
# ------
# Given the context information and not prior knowledge, answer the query.
# {instructions_str}
# Query: {query_str}
# Answer: """

SOURCE_QA_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "{instructions_str}\n"
    "Query: {query_str}\n"
    "Response:\n"
)

# SOURCE_REFINE_TEMPLATE = PromptTemplate(
#     "Please provide an answer based solely on the provided sources. "
#     "When referencing information from a source, "
#     "cite the appropriate source(s) using their corresponding numbers. "
#     "Every answer should include at least one source source. "
#     "Only cite a source when you are explicitly referencing it. "
#     "If none of the sources are helpful, you should indicate that. "
#     "For example:\n"
#     "Source 1:\n"
#     "The sky is red in the evening and blue in the morning.\n"
#     "Source 2:\n"
#     "Water is wet when the sky is red.\n"
#     "Query: When is water wet?\n"
#     "Answer: Water will be wet when the sky is red [2], "
#     "which occurs in the evening [1].\n"
#     "Now it's your turn. "
#     "We have provided an existing answer: {existing_answer}"
#     "Below are several numbered sources of information. "
#     "Use them to refine the existing answer. "
#     "If the provided sources are not helpful, you will repeat the existing answer."
#     "\nBegin refining!"
#     "\n------\n"
#     "{context_msg}"
#     "\n------\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )

DEFAULT_TONE_NAME = "a professional job seeker"
DEFAULT_FORMAT = """
{
  "data": [
    {
        "filename": "filename_1.txt",
        "lines": [<start_line_num>, <end_line_num>],
        "question": "Question 1"
        "answer": "Answer 1"
    }
  ]
}
""".strip()

INSTRUCTIONS_PROMPT = """
Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:
{class_string}
""".strip()

PROMPT_TEMPLATE = """
{general_query_str}
Answers must have complete details based from source lines in the style of {tone_name}.
Example response format:
{format}
""".strip()

DEFAULT_QUERY = """
Generate diverse questions and answers that an employer can have for a job interview based on provided context and schema.
""".strip()


class SourcesGenerator:
    def __init__(self, data_path: str, llm: Ollama | LLM = Ollama(model="llama3.1"), chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm
        self.nodes = self.load_and_parse_data(data_path)

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

    def prepend_line_numbers(self, text):
        lines = text.splitlines()
        numbered_lines = [
            f"line {i + 1}: {line}" for i, line in enumerate(lines)]
        return "\n".join(numbered_lines)

    def generate(self, context_text: str, query: str = DEFAULT_QUERY, tone_name: str = DEFAULT_TONE_NAME, format: str = DEFAULT_FORMAT) -> QueryData:
        qa_prompt_tmpl = PromptTemplate(SOURCE_QA_TEMPLATE)
        general_query_str = PROMPT_TEMPLATE.format(
            general_query_str=query,
            tone_name=tone_name,
            format=format,
        )
        instructions_str = INSTRUCTIONS_PROMPT.format(
            class_string=class_to_string(QueryData),
        )

        response = self.llm.structured_predict(
            QueryData,
            qa_prompt_tmpl,
            context_str=context_text,
            instructions_str=instructions_str,
            query_str=general_query_str,
            llm_kwargs={
                "options": {"temperature": 0},
            },
        )
        return response

    def process(self, query: str = DEFAULT_QUERY) -> Generator[QueryData, None, None]:
        for node in self.nodes:
            nodes_to_parse = [node]
            texts_with_line_numbers = [
                f"File name: {node.metadata['file_name']}\n{
                    self.prepend_line_numbers(node.text)}"
                for node in nodes_to_parse
            ]

            logger.log("Parsed nodes:", len(texts_with_line_numbers),
                       colors=["DEBUG", "SUCCESS"])
            context_text = "\n\n".join(texts_with_line_numbers)

            response = self.generate(context_text, query)

            yield response
