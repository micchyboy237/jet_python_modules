# Simple RAG Application

This project implements a basic Retrieval-Augmented Generation (RAG) system using LangChain.

## Setup

1. Clone the repo and navigate to the directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate it: `source venv/bin/activate` (Unix) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Set your OpenAI API key: `export OPENAI_API_KEY=your_api_key_here`
6. Run the app: `python main.py`

## Usage

- Modify `documents` and `query` in `main.py` as needed.
- Extend by adding persistent storage or advanced retrieval in `rag.py`.

## Notes

- This is for educational purposes; production use requires robust error handling.
- Adapt for other LLM providers by changing configs in `config.py`.
