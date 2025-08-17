import redis
import json
from typing import List, Dict, Any
from jet.logger import logger


def connect_redis(redis_url: str = "redis://localhost:6379") -> redis.Redis:
    """Connect to Redis instance."""
    try:
        client = redis.Redis.from_url(redis_url, decode_responses=True)
        client.ping()  # Test connection
        logger.info("Successfully connected to Redis")
        return client
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {str(e)}")
        raise


def get_redis_keys(client: redis.Redis, prefix: str = "memory*") -> List[str]:
    """Retrieve all keys matching the prefix."""
    try:
        keys = client.keys(prefix)
        logger.info(f"Found {len(keys)} keys with prefix '{prefix}'")
        return keys
    except Exception as e:
        logger.error(f"Error retrieving keys: {str(e)}")
        return []


def get_redis_data(client: redis.Redis, key: str) -> Dict[str, Any]:
    """Retrieve data for a specific key."""
    try:
        data = client.hgetall(key)
        logger.success(
            f"Retrieved data for key {key}: {json.dumps(data, indent=2)}")
        return data
    except Exception as e:
        logger.error(f"Error retrieving data for key {key}: {str(e)}")
        return {}


def check_redis_data() -> None:
    """Verify data stored in RedisMemory."""
    client = connect_redis()
    keys = get_redis_keys(client, prefix="memory*")

    expected_contents = [
        "The weather should be in metric units",
        "Meal recipe must be vegan"
    ]
    expected_metadata = [
        {"category": "preferences", "type": "units"},
        {"category": "preferences", "type": "dietary"}
    ]

    for key in keys:
        data = get_redis_data(client, key)
        content = data.get("content", "")
        metadata = json.loads(data.get("metadata", "{}"))

        if content in expected_contents:
            logger.success(f"Found expected content: {content}")
            expected_index = expected_contents.index(content)
            expected_meta = expected_metadata[expected_index]
            if metadata == expected_meta:
                logger.success(
                    f"Metadata matches expected: {json.dumps(expected_meta)}")
            else:
                logger.warning(
                    f"Metadata mismatch for {content}: got {json.dumps(metadata)}, expected {json.dumps(expected_meta)}")
        else:
            logger.warning(f"Unexpected content found: {content}")

    client.close()
