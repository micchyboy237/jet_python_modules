# Reusable constants
from jet.llm.helpers.dataset_utils import EnhancedLabelledRagDataset, EnhancedQADataset
from jet.llm.ollama.base import initialize_ollama_settings
from jet.llm.ollama.constants import OLLAMA_LARGE_CHUNK_OVERLAP, OLLAMA_LARGE_CHUNK_SIZE, OLLAMA_LARGE_LLM_MODEL
from jet.llm.query.retrievers import get_fusion_retriever, setup_retrievers
from jet.logger import logger
from jet.transformers.object import make_serializable
from jet._token.token_utils import get_tokenizer
from llama_index.core.evaluation.dataset_generation import DatasetGenerator
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset, generate_qa_embedding_pairs
from llama_index.core.llama_dataset.rag import LabelledRagDataExample, LabelledRagDataset
from llama_index.core.node_parser.text.sentence import SentenceSplitter
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.schema import Document
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
import sys
import logging
from llama_index.core.evaluation import EvaluationResult, LabelledQADataset
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation import RelevancyEvaluator, generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
import random
import json

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
LLM_MODEL = "llama3.1"
NUM_QUESTIONS_PER_CHUNK = 3

QUESTION_GEN_QUERY = f"""
You are a Job Employer. Your task is to setup {NUM_QUESTIONS_PER_CHUNK} questions
for an upcoming interview. The questions should be relevant to the document.
Restrict the questions to the context information provided.

Format response with numeric list of questions separated by newline.
Example response format:
1. Question 1
2. Question 2
3. Question 3

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


class QADatasetGenerator:
    def __init__(
        self,
        data_path: str,
        num_questions_per_chunk: int = NUM_QUESTIONS_PER_CHUNK,
        llm_model: str = OLLAMA_LARGE_LLM_MODEL,
        chunk_size: int = OLLAMA_LARGE_CHUNK_SIZE,
        chunk_overlap: int = OLLAMA_LARGE_CHUNK_OVERLAP,
    ):
        self.data_path = data_path
        self.num_questions_per_chunk = num_questions_per_chunk
        self.model = llm_model

        # Setup model, tokenizer and embed settings
        self.tokenizer = get_tokenizer(llm_model)
        self.llm_settings = initialize_ollama_settings({
            "llm_model": self.model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        })

        # Setup documents and nodes
        self.reader = SimpleDirectoryReader(data_path, required_exts=[".md"])
        self.documents = self.reader.load_data()
        self.nodes = self._split_documents()

        self.llm_settings.node_parser.get_nodes_from_documents

        # Create index
        self.vector_index = VectorStoreIndex(
            embed_model=self.llm_settings.embed_model,
            nodes=self.nodes,
            show_progress=True,
        )

        # LLM and evaluator
        self.llm = self.llm_settings.llm
        # self.evaluator = RelevancyEvaluator(llm=self.llm)

    def _split_documents(self) -> list[Document]:
        # create parser and parse documents into nodes
        # parser = SentenceSplitter(
        #     chunk_size=chunk_size,
        #     chunk_overlap=chunk_overlap,
        #     tokenizer=self.tokenizer.encode,
        # )
        nodes = self.llm_settings.node_parser.get_nodes_from_documents(
            self.documents, show_progress=True)

        return [
            Document(
                id=node.node_id,
                text=node.get_content(),
                metadata=node.metadata,
            ) for node in nodes
        ]

    def generate_dataset(self) -> "EnhancedLabelledRagDataset":
        question_template = PromptTemplate(QUESTION_GENERATION_PROMPT)
        data_generator = RagDatasetGenerator.from_documents(
            self.nodes,
            llm=self.llm,
            num_questions_per_chunk=self.num_questions_per_chunk,
            question_gen_query=QUESTION_GEN_QUERY,
            text_question_template=question_template,
            show_progress=True,
        )
        rag_dataset = data_generator.generate_questions_from_nodes()
        query_engine = self.vector_index.as_query_engine()
        return EnhancedLabelledRagDataset(
            rag_dataset,
            query_engine=query_engine,
            llm_model=self.model,
        )

    def generate_questions(self) -> list[str]:
        question_template = PromptTemplate(QUESTION_GENERATION_PROMPT)
        data_generator = DatasetGenerator.from_documents(
            self.nodes,
            llm=self.llm,
            num_questions_per_chunk=self.num_questions_per_chunk,
            question_gen_query=QUESTION_GEN_QUERY,
            text_question_template=question_template,
        )
        questions = data_generator.generate_questions_from_nodes()
        return questions

    def generate_qa_pairs(self) -> EnhancedQADataset:
        qa_generate_prompt_tmpl = QUESTION_GENERATION_PROMPT.format(
            context_str="{context_str}",
            query_str=QUESTION_GEN_QUERY
        )
        qa_dataset = generate_question_context_pairs(
            self.nodes,
            llm=self.llm,
            num_questions_per_chunk=self.num_questions_per_chunk,
            qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
        )

        return EnhancedQADataset(
            qa_dataset,
            self.vector_index,
            llm_model=self.model,
        )

    def evaluate_dataset(self, qa_dataset: EmbeddingQAFinetuneDataset, similarity_k: int = 10):
        # Define the retriever
        # retriever = self.vector_index.as_retriever(similarity_top_k=2)
        retrievers = setup_retrievers(
            self.vector_index, similarity_k, similarity_k)
        fusion_retriever = get_fusion_retriever(
            retrievers, FUSION_MODES.RELATIVE_SCORE, similarity_k)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=fusion_retriever
        )

        # try it out on a sample query
        sample_id, sample_query = list(qa_dataset.queries.items())[0]
        sample_expected = qa_dataset.relevant_docs[sample_id]

        eval_result = retriever_evaluator.evaluate(
            sample_query, sample_expected)
        print(eval_result)

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
