import redis
import json
from typing import List, Dict, Any, Optional, Union
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


def get_redis_keys(host: str = "localhost", port: int = 6379, prefix: str = "*") -> List[str]:
    """Retrieve all keys matching the prefix."""
    redis_url = f"redis://{host}:{port}"
    client = None
    try:
        client = connect_redis(redis_url)
        keys = client.keys(prefix)
        logger.info(f"Found {len(keys)} keys with prefix '{prefix}'")
        return keys
    except Exception as e:
        logger.error(f"Error retrieving keys: {str(e)}")
        return []
    finally:
        if client is not None:
            client.close()


def get_redis_data_by_key(
    key: Union[str, List[str]],
    *,
    host: str = "localhost",
    port: int = 6379,
) -> Dict[str, Any]:
    """Retrieve data for specific key(s) with optional host and port."""
    redis_url = f"redis://{host}:{port}"
    client = None
    result = {}
    try:
        client = connect_redis(redis_url)
        # Normalize key to a list
        keys = [key] if isinstance(key, str) else key
        for k in keys:
            if client.type(k) == "hash":
                data = client.hgetall(k)
                logger.success(
                    f"Retrieved hash data for key {k}: {json.dumps(data, indent=2)}")
            else:
                data_str = client.get(k)
                if data_str is None:
                    logger.warning(f"No data found for key {k}")
                    data = {}
                else:
                    try:
                        data = json.loads(data_str)
                        logger.success(
                            f"Retrieved string data for key {k}: {json.dumps(data, indent=2)}")
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Invalid JSON data for key {k}: {str(e)}")
                        data = {}
            result[k] = data
        return result
    except redis.ConnectionError:
        logger.error(f"Failed to connect to Redis for key(s) {key}")
        return {k: {} for k in keys} if isinstance(key, list) else {key: {}}
    except Exception as e:
        logger.error(f"Error retrieving data for key(s) {key}: {str(e)}")
        return {k: {} for k in keys} if isinstance(key, list) else {key: {}}
    finally:
        if client is not None:
            client.close()


def get_redis_data(
    *,
    key: Optional[Union[str, List[str]]] = None,
    host: str = "localhost",
    port: int = 6379,
) -> Dict[str, Any]:
    """Retrieve data stored in Redis with optional host, port, and specific key(s)."""
    redis_url = f"redis://{host}:{port}"
    client = connect_redis(redis_url)
    result = {"keys": {}, "errors": []}

    try:
        keys = key if isinstance(key, list) else [key] if isinstance(
            key, str) else get_redis_keys(host=host, port=port, prefix="*")

        for k in keys:
            data = get_redis_data_by_key(k, host=host, port=port)
            if not data[k]:
                result["errors"].append(f"No data found for key {k}")
                continue

            result["keys"][k] = data[k]
            logger.success(
                f"Processed data for key {k}: {json.dumps(data[k], indent=2)}")

    except Exception as e:
        logger.error(f"Error processing Redis data: {str(e)}")
        result["errors"].append(f"General error: {str(e)}")
    finally:
        client.close()

    return result


def clear_redis(
    *,
    key: Optional[Union[str, List[str]]] = None,
    host: str = "localhost",
    port: int = 6379,
) -> Dict[str, Any]:
    """Clear specific key(s) or all keys in Redis with optional host, port, and key(s)."""
    redis_url = f"redis://{host}:{port}"
    client = None
    result = {"deleted_keys": [], "errors": []}

    try:
        client = connect_redis(redis_url)
        if key is None:
            result["deleted_keys"] = get_redis_keys(
                host=host, port=port, prefix="*")
            client.flushdb()
            logger.success(
                f"Cleared all keys in Redis database: {result['deleted_keys']}")
        else:
            keys = [key] if isinstance(key, str) else key
            deleted = client.delete(*keys)
            if deleted > 0:
                logger.success(f"Deleted {deleted} key(s): {keys}")
                result["deleted_keys"] = keys
            else:
                logger.warning(f"No keys deleted for: {keys}")
                result["errors"].append(f"No keys found to delete: {keys}")
    except redis.ConnectionError:
        logger.error(
            f"Failed to connect to Redis for clearing key(s) {key or 'all'}")
        result["errors"].append(
            f"Connection error while clearing key(s) {key or 'all'}")
    except Exception as e:
        logger.error(f"Error clearing Redis key(s) {key or 'all'}: {str(e)}")
        result["errors"].append(f"General error: {str(e)}")
    finally:
        if client is not None:
            client.close()

    return result
