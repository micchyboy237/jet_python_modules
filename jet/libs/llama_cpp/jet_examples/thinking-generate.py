from openai import OpenAI

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

prompt = "Think step by step before answering. Show your reasoning clearly.\n\nQuestion: why is the sky blue"

response = client.chat.completions.create(
    model='deepseek-r1',
    messages=[{'role': 'user', 'content': prompt}],
    stream=False,
    temperature=0.7,
)

full_response = response.choices[0].message.content
thinking = ""
answer = ""

# Heuristic split: assume thinking ends before final answer
lines = full_response.strip().split('\n')
final_answer_idx = -1
for i, line in enumerate(lines):
    if any(phrase in line.lower() for phrase in ["answer:", "final", "so,", "therefore"]):
        final_answer_idx = i
        break

if final_answer_idx != -1:
    thinking = "\n".join(lines[:final_answer_idx]).strip()
    answer = "\n".join(lines[final_answer_idx:]).strip()
else:
    thinking = full_response
    answer = ""

print('Thinking:\n========\n\n' + thinking)
print('\nResponse:\n========\n\n' + answer)