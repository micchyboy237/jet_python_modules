# from ollama import chat

# messages = [
#   {
#     'role': 'user',
#     'content': 'Why is the sky blue?',
#   },
# ]

# for part in chat('gemma3', messages=messages, stream=True):
#   print(part['message']['content'], end='', flush=True)


from ollama import Client

client = Client(host='http://localhost:11435')

messages = [
  {
    'role': 'user',
    'content': 'Write a Python function to calculate the factorial of a non-negative integer using recursion. Include error handling for negative inputs.',
  },
]

for part in client.chat(model='deepseek-coder-v2:16b-lite-instruct-q3_K_M', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)