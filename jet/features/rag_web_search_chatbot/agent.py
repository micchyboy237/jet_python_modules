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

# Retriever tool
retriever = get_retriever()
rag_tool = retriever

# Web search tool
web_search = get_web_search_tool()
tools = [web_search, rag_tool]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# State for graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]


# Router node: Decide tool
def router(state):
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    decision = chain.invoke({"question": state["messages"][-1].content})
    if "RAG" in decision.upper():
        tool = rag_tool
    elif "WEB_SEARCH" in decision.upper():
        tool = web_search
    else:
        tool = None
    if tool:
        return {"messages": [HumanMessage(content=f"Use {tool.name} for this query.")]}
    return {"messages": [AIMessage(content="I don't know how to handle this.")]}


# Generation node
def generate(state):
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
    context = ""  # Placeholder: Extend to include retrieved docs
    chain = rag_prompt | llm | StrOutputParser()
    response = chain.invoke(
        {"context": context, "question": state["messages"][-1].content})
    return {"messages": [AIMessage(content=response)]}


# Tool node
tool_node = ToolNode(tools)

# Graph
workflow = StateGraph(state_schema=AgentState)
workflow.add_node("router", router)
workflow.add_node("tools", tool_node)
workflow.add_node("generate", generate)
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router", lambda s: "tools" if "Use" in s["messages"][-1].content else END, {"tools": "tools", END: "generate"})
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
