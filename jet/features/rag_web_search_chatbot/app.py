import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from jet.features.rag_web_search_chatbot.agent import run_agent

load_dotenv(dotenv_path=os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".env"))
st.title("Agentic RAG Chatbot with Ollama")
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.button("Clear Chat History"):
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(
            content=m["content"]) for m in st.session_state.messages[:-1]]
        response = run_agent(prompt, history)
        st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response})
