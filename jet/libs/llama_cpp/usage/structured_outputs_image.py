from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from openai import OpenAI

class Object(BaseModel):
    name: str
    confidence: float
    attributes: str

class ImageDescription(BaseModel):
    summary: str
    objects: list[Object]
    scene: str
    colors: list[str]
    time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night']
    setting: Literal['Indoor', 'Outdoor', 'Unknown']
    text_content: str | None = None

client = OpenAI(base_url="http://shawn-pc.local:8080/v1", api_key="sk-1234")

path = input('Enter the path to your image: ')
path = Path(path)
if not path.exists():
    raise FileNotFoundError(f'Image not found at: {path}')

with open(path, "rb") as image_file:
    image_data = image_file.read()

response = client.chat.completions.create(
    model='gemma3',
    messages=[
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'Analyze this image and return a detailed JSON description including objects, scene, colors and any text detected. If you cannot determine certain details, leave those fields empty.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_data}'}}  # Note: llama.cpp may require base64
            ]
        }
    ],
    response_format={'type': 'json_object', 'schema': ImageDescription.model_json_schema()},
    temperature=0,
)

image_analysis = ImageDescription.model_validate_json(response.choices[0].message.content)
print(image_analysis)