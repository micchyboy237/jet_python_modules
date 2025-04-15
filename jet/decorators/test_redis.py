import unittest
import time
from jet.decorators.redis_wrappers import redis_cache


@redis_cache(ttl=2)
def test_fn(x):
    return {"value": x * 2}


class TestRedisCache(unittest.TestCase):
    def test_caching_behavior(self):
        sample = 5
        expected = {"value": 10}

        result1 = test_fn(sample)
        time.sleep(1)
        result2 = test_fn(sample)
        time.sleep(2)
        result3 = test_fn(sample)

        self.assertEqual(result1, expected)
        self.assertEqual(result2, expected)
        self.assertEqual(result3, expected)  # New result after cache expires


if __name__ == "__main__":
    unittest.main()
