from typing import List, Dict
from jet.adapters.llama_cpp.llm import LlamacppLLM

model = "qwen3-instruct-2507:4b"
client = LlamacppLLM(model=model, verbose=True)

def generate_seed_entities(texts: List[str]) -> List[Dict[str, str]]:
    """
    Use an LLM to auto-generate entity candidates and possible labels from text.
    """
    joined_text = "\n".join(texts[:5])  # sample subset to limit cost

    prompt = f"""
    Extract named entities from the following text.
    For each entity, guess its category (label).
    Return JSON array of objects with 'text' and 'label' fields.

    TEXT:
    {joined_text}
    """

    response_stream = client.chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=400,
        stream=True
    )

    response = ""
    for chunk in response_stream:
        response += chunk

    try:
        import json
        content = response
        entities = json.loads(content)
        return entities
    except Exception as e:
        print("Failed to parse LLM response:", e)
        return []

if __name__ == "__main__":
    sample_texts = [
        "Apple Inc. is headquartered in Cupertino, California.",
        "Elon Musk announced a new Tesla factory in Shanghai.",
        "The Eiffel Tower is a famous landmark in Paris, France.",
        "Google was founded by Larry Page and Sergey Brin.",
        "NASA launched the Artemis I mission from Kennedy Space Center."
    ]

    entities = generate_seed_entities(sample_texts)
    print("Generated entities:")
    for ent in entities:
        print(f"  - {ent['text']}: {ent['label']}")
