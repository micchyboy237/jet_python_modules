from typing import Optional
import httpx
from httpx._types import HeaderTypes
from jet.transformers.object import make_serializable
from pydantic.main import BaseModel
from shared.setup.api_config import api_request_headers


# Create an HTTPX Client with global headers
class HttpxClient:
    def __init__(self, headers: Optional[HeaderTypes] = {}):
        self.client = httpx.Client(
            headers={
                # **settings['request']['headers'],
                **api_request_headers,
                **headers
            }
        )

    def get(self, url: str):
        return self.client.get(url, timeout=300.0)

    def post(self, url: str, json: dict | BaseModel):
        serialized_json = make_serializable(json)
        return self.client.post(url, json=serialized_json, timeout=300.0)


if __name__ == "__main__":
    # Instantiate the client with global headers
    http_client = HttpxClient()

    # Example usage:
    response = http_client.get("http://example.com/api")
    print(response.json())
