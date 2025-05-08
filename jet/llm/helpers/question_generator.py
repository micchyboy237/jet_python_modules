from typing import Generator
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.ollama.constants import OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_LLM_MODEL
from jet.llm.query.retrievers import get_fusion_retriever, setup_retrievers
from jet.logger import logger
from jet.multiprocess.work_manager import WorkManager
from jet.transformers.object import make_serializable
from jet.token.token_utils import get_tokenizer
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import BaseNode, Document, MetadataMode
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
import sys
import logging
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator, generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
import random
import json

from tqdm.asyncio import tqdm_asyncio

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
LLM_MODEL = "llama3.1"
NUM_QUESTIONS_PER_CHUNK = 3

QUESTION_GEN_QUERY = f"""
You are a Job Employer. Your task is to setup multiple questions
for an upcoming interview. Each question should cover a part of the document. Generated questions should be complete so that its answers cover all context information provided.

Format response with numeric list of questions separated by newline.
Example response format:
1. Question 1
...continue

Output only the generated questions without any explanations.
"""
QUESTION_GENERATION_PROMPT = """\
Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge.
generate only questions based on the below query.
Query: {query_str}
"""


class QuestionGenerator:
    def __init__(
        self,
        data_path,
        num_questions_per_chunk: int = NUM_QUESTIONS_PER_CHUNK,
        llm_model: str = OLLAMA_LARGE_LLM_MODEL,
        chunk_size: int = OLLAMA_LARGE_CHUNK_SIZE,
        chunk_overlap: int = OLLAMA_LARGE_CHUNK_OVERLAP,
    ):
        self.data_path = data_path
        self.num_questions_per_chunk = num_questions_per_chunk
        self.model = llm_model
        self.tokenizer = get_tokenizer(llm_model)
        self.llm_settings = initialize_ollama_settings({
            "llm_model": self.model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })
        self.reader = SimpleDirectoryReader(data_path, required_exts=[".md"])
        self.documents = self.reader.load_data()
        self.nodes = self._initialize_nodes(
            self.documents, chunk_size, chunk_overlap)
        self.vector_index_obj = VectorStoreIndex(
            embed_model=self.llm_settings.embed_model,
            nodes=self.documents,
            show_progress=True,
        )
        self.vector_index = self.vector_index_obj.from_documents(
            self.documents,
            embed_model=self.llm_settings.embed_model,
        )
        self.llm = self.llm_settings.llm
        self.evaluator = RelevancyEvaluator(llm=self.llm)

    def _initialize_nodes(
        self,
        documents: list[Document],
        chunk_size: int,
        chunk_overlap: int,
    ):
        # create parser and parse documents into nodes
        parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=self.tokenizer.encode,
        )
        nodes = parser(documents)
        return nodes

    def generate_questions(self, nodes: list[BaseNode] = []) -> list[str]:
        nodes = nodes or self.nodes
        results = []
        for node in nodes:
            question_template = PromptTemplate(QUESTION_GENERATION_PROMPT)
            data_generator = DatasetGenerator(
                [node],
                metadata_mode=MetadataMode.ALL,
                llm=self.llm,
                num_questions_per_chunk=self.num_questions_per_chunk,
                question_gen_query=QUESTION_GEN_QUERY,
                text_question_template=question_template,
            )
            questions = data_generator.generate_questions_from_nodes()
            # Filter questions to include only those starting with a number
            filtered_questions = [q for q in questions if q.endswith("?")]
            results.extend(filtered_questions)
        return results

    def parallel_generate_questions(self) -> list[str]:
        work_callbacks = [lambda: self.generate_questions(
            [node]) for node in self.nodes]
        work_manager = WorkManager(
            work_callbacks=work_callbacks, num_threads=2)
        work_manager.start_threads()
        return []

    def evaluate_questions(self, questions) -> Generator[EvaluationResult, None, None]:
        for question in questions:
            query_engine = self.vector_index.as_query_engine()
            response_vector = query_engine.query(question)
            eval_result = self.evaluator.evaluate_response(
                query=question, response=response_vector
            )
            yield eval_result

    @staticmethod
    def display_eval_df(query, response, eval_result):
        display_jet_source_nodes(query, response.source_nodes)
        logger.newline()
        logger.info("Eval Results:")
        items = [(key, result) for key, result in eval_result.model_dump(
        ).items() if result is not None]
        for key, result in items:
            if key == 'passing':
                logger.log(f"{key.title()}:", "Passed" if result else "Failed", colors=[
                           "DEBUG", "SUCCESS" if result else "ERROR"])
            elif key == 'invalid_result':
                logger.log(f"{key.title()}:", "Valid" if not result else "Invalid", colors=[
                           "DEBUG", "SUCCESS" if not result else "ERROR"])
            else:
                logger.log(f"{key.title()}:", result,
                           colors=["DEBUG", "SUCCESS"])


def main():
    evaluator = QuestionGenerator(
        data_path=DATA_PATH,
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        llm_model=LLM_MODEL,
    )
    questions = evaluator.generate_questions()

    logger.newline()
    logger.info("Generated eval questions:")
    logger.success(json.dumps(make_serializable(questions), indent=2))

    evaluator.evaluate_questions(questions)

    qa_dataset = evaluator.generate_dataset()
    logger.newline()
    logger.info("Generated QA dataset pairs:")
    logger.success(json.dumps(make_serializable(qa_dataset), indent=2))

    evaluator.evaluate_dataset(qa_dataset)

    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
