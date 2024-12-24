# Define a TypedDict for Redis configuration
from typing import TypedDict


class RedisConfigParams(TypedDict, total=False):
    host: str
    port: int
    db: int
    max_connections: int
