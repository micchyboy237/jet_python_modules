# Reusable constants
from jet.llm.ollama import initialize_ollama_settings
from jet.llm.ollama.base import Ollama
from jet.logger import logger
from jet.transformers import make_serializable
from script_utils import display_source_nodes
import sys
import logging
from llama_index.core.evaluation import EvaluationResult
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.evaluation import DatasetGenerator, RelevancyEvaluator, generate_question_context_pairs
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Response
import random
import json

DATA_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/summaries"
NUM_QUESTIONS_PER_CHUNK = 3
LLM_MODEL = "llama3.1"
QUESTION_GEN_QUERY = f"""
You are a Job Employer. Your task is to setup {NUM_QUESTIONS_PER_CHUNK} questions
for an upcoming interview. The questions should be relevant to the document.
Restrict the questions to the context information provided.
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


class QuestionGenerationEvaluator:
    def __init__(self, data_path, num_questions_per_chunk, llm_model):
        self.data_path = data_path
        self.num_questions_per_chunk = num_questions_per_chunk
        self.llm_model = llm_model
        self._initialize_settings()
        self.reader = SimpleDirectoryReader(data_path, required_exts=[".md"])
        self.documents = self.reader.load_data()
        self.vector_index = VectorStoreIndex.from_documents(self.documents)
        self.gpt4 = Ollama(
            temperature=0, model=self.llm_model,
            request_timeout=300.0, context_window=4096
        )
        self.evaluator = RelevancyEvaluator(llm=self.gpt4)

    def _initialize_settings(self):
        initialize_ollama_settings()

    def generate_dataset(self):
        question_template = PromptTemplate(QUESTION_GENERATION_PROMPT)
        data_generator = DatasetGenerator.from_documents(
            self.documents,
            num_questions_per_chunk=self.num_questions_per_chunk,
            question_gen_query=QUESTION_GEN_QUERY,
            text_question_template=question_template,
        )
        qa_dataset = generate_question_context_pairs(
            self.documents, llm=gpt4, num_questions_per_chunk=self.num_questions_per_chunk, qa_generate_prompt_tmpl=QUESTION_GENERATION_PROMPT
        )
        questions = data_generator.generate_questions_from_nodes()
        return random.sample(questions, 5)

    def generate_questions(self):
        question_template = PromptTemplate(QUESTION_GENERATION_PROMPT)
        data_generator = DatasetGenerator.from_documents(
            self.documents,
            num_questions_per_chunk=self.num_questions_per_chunk,
            question_gen_query=QUESTION_GEN_QUERY,
            text_question_template=question_template,
        )
        questions = data_generator.generate_questions_from_nodes()
        return random.sample(questions, 5)

    def evaluate_questions(self, questions):
        results = []
        for question in questions:
            query_engine = self.vector_index.as_query_engine()
            response_vector = query_engine.query(question)
            eval_result = self.evaluator.evaluate_response(
                query=question, response=response_vector
            )
            self.display_eval_df(question, response_vector, eval_result)
            results.append(eval_result)
        return results

    @staticmethod
    def display_eval_df(query, response, eval_result):
        display_source_nodes(query, response.source_nodes)
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
    evaluator = QuestionGenerationEvaluator(
        data_path=DATA_PATH,
        num_questions_per_chunk=NUM_QUESTIONS_PER_CHUNK,
        llm_model=LLM_MODEL,
    )
    questions = evaluator.generate_questions()

    logger.newline()
    logger.info("Generated eval questions:")
    logger.success(json.dumps(make_serializable(questions), indent=2))

    evaluator.evaluate_questions(questions)
    logger.info("\n\n[DONE]", bright=True)


if __name__ == "__main__":
    main()
