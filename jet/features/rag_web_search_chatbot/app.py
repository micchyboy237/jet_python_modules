import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from .agent import run_agent

# Load environment variables from the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(
    os.path.dirname(os.path.dirname(__file__))), ".env"))

st.title("Agentic RAG Chatbot with Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Add clear chat history button
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

if __name__ == "__main__":
    # Resolve the project root dynamically
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.insert(0, project_root)

    # Run Streamlit with the resolved path to app.py
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", os.path.abspath(__file__)]
    sys.exit(stcli.main())
