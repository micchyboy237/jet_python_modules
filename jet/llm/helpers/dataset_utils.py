from typing import Generator
from jet.llm.query.retrievers import setup_index
from jet.logger import logger
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.relevancy import RelevancyEvaluator
from llama_index.core.evaluation.retrieval.base import RetrievalEvalResult
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator
from llama_index.core.indices.base import BaseIndex, IndexType
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.llama_dataset.legacy.embedding import EmbeddingQAFinetuneDataset, generate_qa_embedding_pairs
from llama_index.core.schema import TextNode
from tqdm import tqdm
from jet.llm.ollama.base import Ollama, OllamaEmbedding
from jet.llm.ollama.constants import OLLAMA_BASE_EMBED_URL, OLLAMA_LARGE_EMBED_MODEL, OLLAMA_LARGE_LLM_MODEL, OLLAMA_SMALL_LLM_MODEL
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.evaluation.answer_relevancy import AnswerRelevancyEvaluator
from llama_index.core.evaluation.context_relevancy import ContextRelevancyEvaluator
from llama_index.core.llama_dataset.base import BaseLlamaDataExample, BaseLlamaExamplePrediction, BaseLlamaPredictionDataset
from llama_index.core.llama_dataset.rag import LabelledRagDataExample, LabelledRagDataset
from llama_index.core.llama_dataset import (
    LabelledRagDataExample,
    CreatedByType,
    CreatedBy,
)
from llama_index.core.prompts.base import PromptTemplate

ANSWER_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the provided response match the subject matter of the user's query?\n"
    "2. Does the provided response attempt to address the focus or perspective "
    "on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above  "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_int_score>\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)
CONTEXT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the retrieved context from the document sources are relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the retrieved context match the subject matter of the user's query?\n"
    "2. Can the retrieved context be used exclusively to provide a full answer to the user's query?\n"
    "Each question above is worth 2 points, where partial marks are allowed and encouraged. Provide detailed feedback on the response "
    "according to the criteria questions previously mentioned. "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the floating number representing the total score assigned to the response'\n\n"
    "Example feedback format:\nFeedback:\n<generated_feedback>\n\n[RESULT] <total_score:.2f>\n\n"
    "Query: \n {query_str}\n"
    "Context: \n {context_str}\n"
    "Feedback:"
)


class EnhancedLabelledRagDataset:
    rag_dataset: LabelledRagDataset
    query_engine: BaseQueryEngine
    prediction_dataset: BaseLlamaPredictionDataset
    prediction_batch_size: int
    data: list[LabelledRagDataExample] = []

    def __init__(
        self,
        rag_dataset: LabelledRagDataset,
        query_engine: BaseQueryEngine,
        *,
        llm_model: str = OLLAMA_LARGE_LLM_MODEL,
        eval_answer_llm_model=OLLAMA_SMALL_LLM_MODEL,
        eval_context_llm_model=OLLAMA_LARGE_LLM_MODEL,
        prediction_batch_size: int = 20,
    ):
        self.rag_dataset = rag_dataset
        self.data = rag_dataset.examples
        self.queries = [d.query for d in self.data]
        self.query_engine = query_engine
        self.llm_model = llm_model
        self.prediction_batch_size = prediction_batch_size

        # Setup eval model judges
        self.yes_no_evaluator = RelevancyEvaluator(
            llm=Ollama(model=eval_answer_llm_model),
        )
        self.answer_relevancy_model = AnswerRelevancyEvaluator(
            llm=Ollama(model=eval_answer_llm_model),
            eval_template=ANSWER_EVAL_TEMPLATE,
        )
        self.context_relevancy_model = ContextRelevancyEvaluator(
            llm=Ollama(model=eval_context_llm_model),
            eval_template=CONTEXT_EVAL_TEMPLATE,
        )

    def evaluate_question(self, question) -> "EvaluationResult":
        response_vector = self.query_engine.query(question)
        eval_result = self.yes_no_evaluator.evaluate_response(
            query=question, response=response_vector
        )
        return eval_result

    def evaluate_queries(self) -> "list[EvaluationResult]":
        eval_results = []
        for query in self.queries:
            response_vector = self.query_engine.query(query)
            eval_result = self.yes_no_evaluator.evaluate_response(
                query=query, response=response_vector
            )
            eval_results.append(eval_result)
        return eval_results

    def evaluate_qa_dataset(self) -> Generator[dict, None, None]:
        if not self.prediction_dataset:
            self.prediction_dataset = self._generate_prediction_dataset()

        predictions = self.prediction_dataset.predictions
        eval_iterator = tqdm(zip(self.data, predictions),
                             total=len(predictions) * self.prediction_batch_size)

        for d, prediction in eval_iterator:
            logger.log("Query:", d.query, colors=["GRAY", "DEBUG"])

            logger.debug("Evaluating answer relevancy...")
            answer_relevancy_result = self.answer_relevancy_model.evaluate(
                query=d.query,
                response=prediction.response,
                sleep_time_in_seconds=1.0,
            )

            logger.debug("Evaluating context relevancy...")
            context_relevancy_result = self.context_relevancy_model.evaluate(
                query=d.query,
                contexts=prediction.contexts,
                sleep_time_in_seconds=1.0,
            )

            # query = d.query
            # query_by = CreatedBy(type=CreatedByType.AI, model_name=self.llm_model)
            # reference_answer = "Yes it is."
            # reference_answer_by = CreatedBy(type=CreatedByType.HUMAN)
            # reference_contexts = ["This is a sample context"]

            # eval_result = LabelledRagDataExample(
            #     query=query,
            #     query_by=query_by,
            #     reference_contexts=reference_contexts,
            #     reference_answer=reference_answer,
            #     reference_answer_by=reference_answer_by,
            # )

            eval_result = {
                "query": d.query,
                "answer_relevancy": {
                    "response": prediction.response,
                    "result": answer_relevancy_result,
                },
                "context_relevancy": {
                    "contexts": prediction.contexts,
                    "result": context_relevancy_result,
                },
            }
            yield eval_result

    def _generate_prediction_dataset(self) -> BaseLlamaPredictionDataset:
        prediction_dataset = self.rag_dataset.make_predictions_with(
            predictor=self.query_engine,
            batch_size=self.prediction_batch_size,
            show_progress=True,
        )
        return prediction_dataset


class EnhancedDataset:
    rag_dataset: LabelledRagDataset
    qa_dataset: EmbeddingQAFinetuneDataset
    index: BaseIndex
    query_engine: BaseQueryEngine
    retriever: BaseRetriever

    def __init__(
        self,
        dataset: LabelledRagDataset | EmbeddingQAFinetuneDataset,
        index: BaseIndex,
    ):
        if isinstance(dataset, EmbeddingQAFinetuneDataset):
            self.qa_dataset = dataset
        else:
            self.rag_dataset = dataset

        self.index = index
        self.query_engine = index.as_query_engine()
        self.retriever = index.as_retriever()


class EnhancedQADataset:
    qa_dataset: EmbeddingQAFinetuneDataset
    index: BaseIndex
    query_engine: BaseQueryEngine
    retriever: BaseRetriever
    # Data
    queries: dict[str, str]  # dict id -> query
    corpus: dict[str, str]  # dict id -> string
    relevant_docs: dict[str, list[str]]  # query id -> list of doc ids
    mode: str = "text"

    def __init__(
        self,
        qa_dataset: EmbeddingQAFinetuneDataset,
        index: BaseIndex,
        *,
        llm_model: str = OLLAMA_LARGE_LLM_MODEL,
        eval_answer_llm_model=OLLAMA_SMALL_LLM_MODEL,
        eval_context_llm_model=OLLAMA_LARGE_LLM_MODEL,
        prediction_batch_size: int = 20,
    ):
        self.index = index
        self.query_engine = index.as_query_engine()
        self.retriever = index.as_retriever()

        self.qa_dataset = qa_dataset
        self.corpus = self.qa_dataset.corpus
        self.queries = self.qa_dataset.queries
        self.relevant_docs = self.qa_dataset.relevant_docs
        self.mode = self.qa_dataset.mode

        self.llm_model = llm_model
        self.prediction_batch_size = prediction_batch_size

        # Setup eval model judges
        self.eval_embed_model = OllamaEmbedding(
            base_url=OLLAMA_BASE_EMBED_URL,
            model_name=OLLAMA_LARGE_EMBED_MODEL,
        )
        self.yes_no_evaluator = RelevancyEvaluator(
            llm=Ollama(model=eval_answer_llm_model),
        )
        self.answer_relevancy_model = AnswerRelevancyEvaluator(
            llm=Ollama(model=eval_answer_llm_model),
            eval_template=ANSWER_EVAL_TEMPLATE,
        )
        self.context_relevancy_model = ContextRelevancyEvaluator(
            llm=Ollama(model=eval_context_llm_model),
            eval_template=CONTEXT_EVAL_TEMPLATE,
        )

    def evaluate_dataset(
        self,
        top_k=10,
    ) -> Generator[RetrievalEvalResult, None, None]:
        corpus = self.corpus
        queries = self.queries
        relevant_docs = self.relevant_docs

        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(
            nodes, embed_model=self.eval_embed_model, show_progress=True
        )
        retriever = index.as_retriever(similarity_top_k=top_k)
        retriever_evaluator = RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )

        # try it out on a sample query
        for id, query in list(queries.items()):
            expected = relevant_docs[id]

            eval_result = retriever_evaluator.evaluate(query, expected)
            yield eval_result

    def evaluate_dataset_2(
        self,
        top_k=10,
    ):
        corpus = self.corpus
        queries = self.queries
        relevant_docs = self.relevant_docs

        nodes = [TextNode(id_=id_, text=text) for id_, text in corpus.items()]
        index = VectorStoreIndex(
            nodes, embed_model=self.eval_embed_model, show_progress=True
        )
        retriever = index.as_retriever(similarity_top_k=top_k)

        eval_results = []
        for query_id, query in tqdm(queries.items()):
            retrieved_nodes = retriever.retrieve(query)
            retrieved_ids = [node.node.node_id for node in retrieved_nodes]
            expected_id = relevant_docs[query_id][0]
            is_hit = expected_id in retrieved_ids  # assume 1 relevant doc

            eval_result = {
                "is_hit": is_hit,
                "retrieved": retrieved_ids,
                "expected": expected_id,
                "query": query_id,
            }
            eval_results.append(eval_result)
        return eval_results
