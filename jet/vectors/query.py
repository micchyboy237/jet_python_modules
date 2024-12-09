# import os
# from llama_index.llms.openai import OpenAI
# from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType
from llama_index.core import VectorStoreIndex, ServiceContext, download_loader
from llama_index.core.prompts import PromptTemplate
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.schema import Document, QueryBundle, NodeWithScore
from llama_index.core.query_engine.transform_query_engine import TransformQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.llms import LLM, ChatMessage


def load_wikipedia_data(pages):
    WikipediaReader = download_loader("WikipediaReader")
    loader = WikipediaReader()
    return loader.load_data(pages=pages, auto_suggest=False, redirect=False)


def create_settings(llm_model, embedding_model, chunk_size, chunk_overlap, base_url):
    # llm = Ollama(model=llm_model, request_timeout=120.0)
    # embed_model = OllamaEmbedding(model=embedding_model)
    # return ServiceContext.from_defaults(
    #     llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap, embed_model=embed_model
    # )
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap
    Settings.embed_model = OllamaEmbedding(
        model_name=embedding_model,
        base_url=base_url,
    )
    Settings.llm = Ollama(
        model=llm_model,
        base_url=base_url,
    )
    return Settings


def create_index(documents):
    return VectorStoreIndex.from_documents(documents)


def create_retriever(index, top_k):
    return index.as_retriever(similarity_top_k=top_k)


def generate_prompt(template, context, question):
    qa_template = PromptTemplate(template)
    return qa_template.format(context_str="\n\n".join(context), query_str=question)


def rerank_nodes(index, query, top_n=3, model="BAAI/bge-reranker-base"):
    retriever = index.as_retriever(similarity_top_k=5)
    nodes = retriever.retrieve(query)
    reranker = FlagEmbeddingReranker(top_n=top_n, model=model)
    query_bundle = QueryBundle(query_str=query)
    ranked_nodes = reranker._postprocess_nodes(
        nodes, query_bundle=query_bundle)
    return ranked_nodes


def hyde_query_transform(index, query):
    hyde = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(
        index.as_query_engine(similarity_top_k=4), hyde)
    return hyde_query_engine.query(query)


def query_model(llm, prompt):
    return llm.complete(prompt)


def query_chat(llm: LLM, messages: list[ChatMessage]):
    return llm.chat(messages)


def main():
    # Configuration
    base_url = "http://localhost:11434"
    llm_model = "llama3.1"
    embedding_model = "nomic-embed-text"
    reranking_model = "BAAI/bge-reranker-base"
    chunk_size = 512
    chunk_overlap = 50
    retriever_top_k = 3
    reranker_top_n = 3

    # Inputs
    pages = ['Emma_Stone', 'La_La_Land', 'Ryan_Gosling']
    prompt_template = (
        "We have provided context information below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
        "Don't give an answer unless it is supported by the context above.\n"
    )

    # Load the documents
    documents = load_wikipedia_data(pages)
    # base_docs = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs"
    # from langchain_community.document_loaders.generic import GenericLoader
    # from langchain_community.document_loaders.parsers import LanguageParser
    # loader = GenericLoader.from_filesystem(
    #     base_docs,
    #     glob="**/*",
    #     suffixes=[".mdx"],
    #     parser=LanguageParser("markdown"),
    # )
    # documents = loader.load()

    service_context = create_settings(
        llm_model, embedding_model, chunk_size, chunk_overlap, base_url)
    index = create_index(documents)
    retriever = create_retriever(index, retriever_top_k)

    questions = [
        "What is the plot of the film that led Emma Stone to win her first Academy Award?",
        "Compare the families of Emma Stone and Ryan Gosling"
    ]

    for question in questions:
        contexts: list[NodeWithScore] = retriever.retrieve(question)
        context_list = [n.get_content() for n in contexts]
        prompt = generate_prompt(prompt_template, context_list, question)
        response = query_model(service_context.llm, prompt)
        print(f"Question: {question}\nResponse: {response}\n")

    rerank_query = "Compare the families of Emma Stone and Ryan Gosling"
    ranked_nodes = rerank_nodes(
        index, rerank_query, top_n=reranker_top_n, model=reranking_model)
    print("Re-ranked Responses:")
    for node in ranked_nodes:
        print(node.get_content())

    hyde_query = "Compare the families of Emma Stone and Ryan Gosling"
    hyde_response = hyde_query_transform(index, hyde_query)
    print("HyDE Query Response:", hyde_response)

    messages = [
        ChatMessage(role="system",
                    content="You are a pirate with a colorful personality."),
        ChatMessage(role="user", content="What is your name?")
    ]
    chat_response = query_chat(service_context.llm, messages)
    print("Chat Response:", chat_response)


if __name__ == "__main__":
    main()
