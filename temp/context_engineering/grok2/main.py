from context_engineer import generate_response

if __name__ == "__main__":
    query = "What is context engineering?"
    docs = [
        "Context engineering manages info for AI.",
        "Prompt engineering is about crafting text inputs.",
        "RAG retrieves documents to augment LLMs."
    ]
    response = generate_response(query, docs)
    print("Generated Response:\n", response)