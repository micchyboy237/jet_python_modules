from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from typing import TypedDict, Annotated, Sequence
import operator
from .retriever import get_retriever
from .tools import get_web_search_tool
from .config import LLM_MODEL, OLLAMA_BASE_URL, GENERATION_PARAMS
from .prompts import RAG_PROMPT, ROUTER_PROMPT

llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL,
                 **GENERATION_PARAMS)
retriever = get_retriever()
web_search = get_web_search_tool()
tools = [web_search]  # Only include web_search as a tool
llm_with_tools = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]


def router(state):
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm_with_tools
    decision = chain.invoke({"question": state["messages"][-1].content})

    if "WEB_SEARCH" in decision.content.upper():
        # Return AIMessage with tool call for web search
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tavily_search_results_json",
                            "args": {"query": state["messages"][-1].content},
                            "id": "web_search_1",
                            "type": "tool_call"
                        }
                    ]
                )
            ]
        }
    elif "RAG" in decision.content.upper():
        return {"messages": [HumanMessage(content="Use RAG retriever for this query.")]}
    return {"messages": [AIMessage(content="I don't know how to handle this.")]}


def generate(state):
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    context = ""
    if "Use RAG retriever" in state["messages"][-1].content:
        query = state["messages"][-2].content  # Get the original question
        retrieved_docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
    chain = rag_prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"context": context, "question": state["messages"][-2].content if "Use RAG retriever" in state["messages"][-1].content else state["messages"][-1].content})
    return {"messages": [AIMessage(content=response)]}


tool_node = ToolNode(tools)
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("router", router)
workflow.add_node("tools", tool_node)
workflow.add_node("generate", generate)
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda s: "tools" if "Use web_search" in s["messages"][-1].content else "generate",
    {"tools": "tools", "generate": "generate"}
)
workflow.add_edge("tools", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()


def run_agent(query, chat_history=None):
    if chat_history:
        messages = chat_history + [HumanMessage(content=query)]
    else:
        messages = [HumanMessage(content=query)]
    result = app.invoke({"messages": messages})
    return result["messages"][-1].content
