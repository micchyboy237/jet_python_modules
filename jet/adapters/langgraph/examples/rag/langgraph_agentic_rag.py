"""LangGraph Agentic RAG with Jet Adapters Integration.

Reuses jet/adapters features:
- ChatLlamaCpp via get_chat_openai() for all LLM nodes
- OpenAIEmbeddings via get_openai_embeddings() for vectorstore
- Centralized config from jet.adapters.llama_cpp.config
- Automatic verbose logging via ChatLogger (no manual prints)
- Persistent Chroma vectorstore
- Inline RAG prompt (no langchain.hub dependency)
"""

import os
from typing import Annotated, Literal, Sequence, TypedDict

from jet.adapters.langchain.factory import get_chat_openai, get_openai_embeddings
from jet.adapters.llama_cpp.config import EMBED_MODEL_LG, LLM_MODEL
from jet.logger import CustomLogger
from jet.logger.config import DEFAULT_LOGGER
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# ---------------------------------------------------------------------------
# Shared Resources (initialized once, reused across all nodes)
# ---------------------------------------------------------------------------

# Single LLM instance for agent, grader, rewriter, generator
# verbose=True enables ChatLogger; enable_thinking=False per requirement
llm = get_chat_openai(
    model=LLM_MODEL,
    temperature=0,
    streaming=True,
    verbose=True,
    enable_thinking=False,
    agent_name="agentic_rag",
)

# Embeddings using LG-optimized model (nomic-embed:1.5, 768 dims)
embeddings = get_openai_embeddings(embed_model=EMBED_MODEL_LG)

# Persistent Chroma vectorstore directory
CHROMA_PERSIST_DIR = os.path.join(
    os.getenv("JET_DATA_DIR", "./data"), "chroma_agentic_rag"
)

logger = CustomLogger(DEFAULT_LOGGER, filename="agentic_rag_init.log")


def _build_vectorstore() -> Chroma:
    """Load existing Chroma DB or build from scratch."""
    if os.path.exists(CHROMA_PERSIST_DIR):
        logger.info("Loading existing Chroma vectorstore from %s", CHROMA_PERSIST_DIR)
        return Chroma(
            collection_name="rag-chroma-local",
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    logger.info("Building new Chroma vectorstore...")
    urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
        "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    ]
    docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma-local",
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info(
        "Vectorstore built with %d chunks, persisted to %s",
        len(doc_splits),
        CHROMA_PERSIST_DIR,
    )
    return vectorstore


vectorstore = _build_vectorstore()
retriever = vectorstore.as_retriever()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

from langchain.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts on LLM agents, prompt engineering, and adversarial attacks on LLMs.",
)
tools = [retriever_tool]

# ---------------------------------------------------------------------------
# State Definition
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Inline RAG Prompt (replaces hub.pull("rlm/rag-prompt"))
# Source: https://smith.langchain.com/hub/rlm/rag-prompt
# ---------------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.",
        ),
        ("human", "{question}\n\nContext:\n{context}"),
    ]
)

# ---------------------------------------------------------------------------
# Graph Nodes (all use shared `llm` instance — no per-call instantiation)
# ---------------------------------------------------------------------------


def grade_documents(state: AgentState) -> Literal["generate", "rewrite"]:
    """Determines whether retrieved documents are relevant to the question."""

    class Grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    llm_with_tool = llm.with_structured_output(Grade)

    prompt = PromptTemplate(
        template=(
            "You are a grader assessing relevance of a retrieved document to a user question.\n\n"
            "Here is the retrieved document:\n\n{context}\n\n"
            "Here is the user question: {question}\n\n"
            "If the document contains keyword(s) or semantic meaning related to the user question, "
            "grade it as relevant.\n"
            "Give a binary score 'yes' or 'no' to indicate whether the document is relevant."
        ),
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool

    messages = state["messages"]
    last_message = messages[-1]
    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    if scored_result.binary_score == "yes":
        return "generate"
    return "rewrite"


def agent(state: AgentState) -> dict:
    """Invokes the agent model to decide retrieval or end."""
    messages = state["messages"]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def rewrite(state: AgentState) -> dict:
    """Transforms the query to produce a better question."""
    messages = state["messages"]
    question = messages[0].content

    msg = [
        HumanMessage(
            content=(
                "Look at the input and try to reason about the underlying semantic intent / meaning.\n\n"
                f"Here is the initial question:\n-------\n{question}\n-------\n\n"
                "Formulate an improved question:"
            )
        )
    ]

    response = llm.invoke(msg)
    return {"messages": [response]}


def generate(state: AgentState) -> dict:
    """Generates final answer from retrieved context."""
    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]
    docs = last_message.content

    rag_chain = RAG_PROMPT | llm | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("agent", agent)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    tools_condition,
    {"tools": "retrieve", END: END},
)
workflow.add_conditional_edges("retrieve", grade_documents)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

graph = workflow.compile()

# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pprint

    inputs = {
        "messages": [
            ("user", "What does Lilian Weng say about the types of agent memory?"),
        ]
    }

    for output in graph.stream(inputs):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")
