from typing import TypedDict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from jet.adapters.langchain.chat_ollama import ChatOllama

class RAGInput(TypedDict):
    query: str
    retriever: FAISS  # Or any BaseRetriever

def rag_query(input_data: RAGInput) -> str:
    """Retrieve relevant chunks and generate response using RAG."""
    query = input_data["query"]
    retriever = input_data["retriever"]

    relevant_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = PromptTemplate(
        template="""Use the following context to answer the question. If the context doesn't contain the answer, say so.

Context: {context}

Question: {query}

Answer:""",
        input_variables=["context", "query"]
    )

    llm = ChatOllama(model="llama3.2", temperature=0)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({"context": context, "query": query})
