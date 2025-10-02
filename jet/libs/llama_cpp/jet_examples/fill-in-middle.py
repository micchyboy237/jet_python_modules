from openai import OpenAI

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")  # Dummy API key
prompt = '''def remove_non_ascii(s: str) -> str:
    """ '''
suffix = """
    return result
"""
response = client.completions.create(
    model="ggml-org/gemma-3-4b-it-GGUF",
    prompt=prompt,
    suffix=suffix,
    max_tokens=128,
    temperature=0,
    top_p=0.9,
    stop=["<EOT>"],
)
print(response.choices[0].text)