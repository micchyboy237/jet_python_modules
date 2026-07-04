from openai import OpenAI

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

def heading(text):
    print(text)
    print('=' * len(text))

messages = [
    {'role': 'user', 'content': 'What is 10 + 23?'},
]

# Map Ollama think levels → prompt strength
think_prompts = {
    'low': "Think briefly before answering.",
    'medium': "Think step by step. Show your reasoning.",
    'high': "Think carefully step by step. Explain every step in detail before giving the final answer."
}

levels = ['low', 'medium', 'high']

for i, level in enumerate(levels):
    system_prompt = think_prompts[level]
    full_messages = [
        {'role': 'system', 'content': system_prompt},
        *messages
    ]

    response = client.chat.completions.create(
        model='gpt-oss:20b',
        messages=full_messages,
        temperature=0,
        stream=False,
    )

    content = response.choices[0].message.content.strip()
    thinking = content
    answer = ""

    # Try to extract final answer
    if "answer" in content.lower():
        parts = content.lower().split("answer")
        thinking = parts[0].strip()
        answer = parts[-1].strip().lstrip(": ")
    else:
        answer = content

    heading(f'Thinking ({level})')
    print(thinking)
    print('\n')
    heading('Response')
    print(answer)
    print('\n')
    if i < len(levels) - 1:
        print('-' * 20)
        print('\n')