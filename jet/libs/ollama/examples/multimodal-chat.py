from ollama import chat, Client

from jet.logger import logger
from jet.transformers.formatters import format_json

# Initialize the client with the new host
client = Client(host='http://localhost:11435')

# from pathlib import Path

# Pass in the path to the image
# path = input('Please enter the path to the image: ')
path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/jet_python_modules/jet/libs/mlx-vlm/examples/images/cats.jpg"

# You can also pass in base64 encoded image data
# img = base64.b64encode(Path(path).read_bytes()).decode()
# or the raw bytes
# img = Path(path).read_bytes()

response = client.chat(
  model='gemma3:4b-it-q4_K_M',
  # model='qwen2.5vl:3b-q4_K_M',
  messages=[
    {
      'role': 'user',
      'content': 'What is in this image? Be concise.',
      'images': [path],
    }
  ],
)

print(response['message']['content'])

logger.gray("Result:")
logger.success(format_json(response))