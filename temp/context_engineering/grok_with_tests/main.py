import os
from context_engineer import generate_response

if __name__ == "__main__":
    # Replace with your actual key if not set in environment
    os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"

    query = "What is context engineering?"
    docs = [
        "Context engineering manages info for AI.",
        "Prompt engineering is about crafting text inputs.",
        "RAG retrieves documents to augment LLMs."
    ]
    response = generate_response(query, docs)
    print("Generated Response:\n", response)