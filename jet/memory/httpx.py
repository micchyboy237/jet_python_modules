from typing import Optional
import httpx
from httpx._types import HeaderTypes
from jet.transformers.object import make_serializable
from pydantic.main import BaseModel
from shared.globals import settings


# Create an HTTPX Client with global headers
class HttpxClient:
    def __init__(self, headers: Optional[HeaderTypes] = {}):
        self.client = httpx.Client(
            headers={
                **settings['request']['headers'],
                **headers
            }
        )

    def get(self, url: str):
        return self.client.get(url)

    def post(self, url: str, json: dict | BaseModel):
        return self.client.post(url, json=make_serializable(json))


if __name__ == "__main__":
    # Instantiate the client with global headers
    http_client = HttpxClient()

    # Example usage:
    response = http_client.get("http://example.com/api")
    print(response.json())
