from typing import List, Optional
from jet.llm.ollama.base import Ollama
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import TextNode
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


class Data(BaseModel):
    question: str = Field(
        description="Short question text answering context information provided.")
    answer: str = Field(
        description="The concise answer to the question given the relevant context.")


class QuestionAnswer(BaseModel):
    data: List[Data]


QA_PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "{instructions_str}\n"
    "Query: {query_str}\n"
    "Response:\n"
)

INSTRUCTIONS_PROMPT = f"""
Return only the generated JSON value without any explanations surrounded by ```json that adheres to the model below:
{class_to_string(QuestionAnswer)}
""".strip()

DEFAULT_QUERY = """Generate real-world diverse questions and answers that an employer can have for a job interview based on provided context and schema.
Example response format:
{
    "data": [
        {
            "question": "Question 1",
            "answer": "Answer 1"
        }
    ]
}
""".strip()


class InterviewQAGenerator:
    def __init__(self, llm: Ollama | LLM = Ollama(model="llama3.1"), chunk_size: int = 1024, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.llm = llm

    def load_and_parse_data(self, data_path: str):
        logger.newline()
        logger.info("Loading data...")
        docs = SimpleDirectoryReader(data_path).load_data(show_progress=True)
        logger.log("All docs:", len(docs), colors=["DEBUG", "SUCCESS"])
        base_nodes = parse_nodes(
            docs, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        logger.log("Parsed nodes:", len(base_nodes),
                   colors=["DEBUG", "SUCCESS"])
        return "\n\n".join([doc.text for doc in base_nodes])

    def generate_questions_and_answers(self, context_text: str, query: str = DEFAULT_QUERY) -> QuestionAnswer:
        qa_prompt = PromptTemplate(QA_PROMPT_TEMPLATE)

        response = self.llm.structured_predict(
            QuestionAnswer,
            qa_prompt,
            context_str=context_text,
            instructions_str=INSTRUCTIONS_PROMPT,
            query_str=query,
            llm_kwargs={
                "options": {"temperature": 0},
            },
        )
        return response

    def process(self, data_path: Optional[str] = None, nodes: Optional[list[TextNode]] = None, query: str = DEFAULT_QUERY) -> QuestionAnswer:
        if data_path:
            context_text = self.load_and_parse_data(data_path)
        elif nodes:
            context_text = "\n\n".join([node.text for node in nodes])
        response = self.generate_questions_and_answers(context_text, query)
        return response


# Usage Example
def main():
    data_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"

    processor = InterviewQAGenerator()
    response = processor.process(data_path)

    logger.newline()
    logger.info("RESPONSE:")
    logger.success(format_json(response))
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
