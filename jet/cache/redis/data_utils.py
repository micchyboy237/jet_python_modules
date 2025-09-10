import redis
import json
from typing import List, Dict, Any, Optional
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


def get_redis_keys(client: redis.Redis, prefix: str = "*") -> List[str]:
    """Retrieve all keys matching the prefix."""
    try:
        keys = client.keys(prefix)
        logger.info(f"Found {len(keys)} keys with prefix '{prefix}'")
        return keys
    except Exception as e:
        logger.error(f"Error retrieving keys: {str(e)}")
        return []


def get_redis_data_by_key(
    key: str,
    *,
    host: str = "localhost",
    port: int = 6379,
) -> Dict[str, Any]:
    """Retrieve data for a specific key with optional host, port, and key."""
    redis_url = f"redis://{host}:{port}"
    client = None
    try:
        client = connect_redis(redis_url)
        # Try hgetall for hash keys (e.g., memory* keys)
        if client.type(key) == "hash":
            data = client.hgetall(key)
            logger.success(
                f"Retrieved hash data for key {key}: {json.dumps(data, indent=2)}")
        else:
            # Try get for string keys (e.g., tavily_search:* keys)
            data_str = client.get(key)
            if data_str is None:
                logger.warning(f"No data found for key {key}")
                return {}
            try:
                data = json.loads(data_str)
                logger.success(
                    f"Retrieved string data for key {key}: {json.dumps(data, indent=2)}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON data for key {key}: {str(e)}")
                return {}
        return data
    except redis.ConnectionError:
        logger.error(f"Failed to connect to Redis for key {key}")
        return {}
    except Exception as e:
        logger.error(f"Error retrieving data for key {key}: {str(e)}")
        return {}
    finally:
        if client is not None:  # Close the connection only if we created it
            client.close()


def get_redis_data(
    *,
    key: Optional[str] = None,
    host: str = "localhost",
    port: int = 6379,
) -> Dict[str, Any]:
    """Verify and retrieve data stored in RedisMemory with optional host, port, and specific key."""
    redis_url = f"redis://{host}:{port}"
    client = connect_redis(redis_url)

    expected_contents = [
        "The weather should be in metric units",
        "Meal recipe must be vegan"
    ]
    expected_metadata = [
        {"category": "preferences", "type": "units"},
        {"category": "preferences", "type": "dietary"}
    ]

    result = {"keys": {}, "errors": []}

    try:
        keys = [key] if key else get_redis_keys(client, prefix="*")

        for key in keys:
            data = get_redis_data_by_key(key, host=host, port=port)
            if not data:
                result["errors"].append(f"No data found for key {key}")
                continue

            # Initialize default values
            content = data.get("content", "")
            metadata = data.get("metadata", {})

            # Handle Tavily search results (stored as JSON strings)
            if key.startswith("tavily_search:"):
                content = data.get("answer", "") or json.dumps(
                    data.get("results", []))
                metadata = {
                    "query": data.get("query", ""),
                    "max_results": data.get("max_results", 5),
                    "search_depth": data.get("search_depth", "advanced")
                }
            else:
                # Try parsing metadata as JSON for non-Tavily keys
                try:
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Invalid metadata JSON for key {key}: {str(e)}")
                    result["errors"].append(f"Invalid metadata for key {key}")
                    continue

            result["keys"][key] = {"content": content, "metadata": metadata}

            # Only validate expected content/metadata for non-Tavily keys
            if not key.startswith("tavily_search:") and content in expected_contents:
                logger.success(f"Found expected content: {content}")
                expected_index = expected_contents.index(content)
                expected_meta = expected_metadata[expected_index]
                if metadata == expected_meta:
                    logger.success(
                        f"Metadata matches expected: {json.dumps(expected_meta)}")
                else:
                    logger.warning(
                        f"Metadata mismatch for {content}: got {json.dumps(metadata)}, expected {json.dumps(expected_meta)}")
                    result["errors"].append(f"Metadata mismatch for key {key}")
            elif not key.startswith("tavily_search:"):
                logger.warning(f"Unexpected content found: {content}")
                result["errors"].append(f"Unexpected content for key {key}")

    except Exception as e:
        logger.error(f"Error processing Redis data: {str(e)}")
        result["errors"].append(f"General error: {str(e)}")
    finally:
        client.close()

    return result
