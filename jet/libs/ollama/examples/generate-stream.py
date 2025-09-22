from ollama import Client

client = Client(host='http://localhost:11435')

for part in client.generate('gemma3', 'Why is the sky blue?', stream=True):
  print(part['response'], end='', flush=True)
