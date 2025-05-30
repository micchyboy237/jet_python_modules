import unittest

from jet.utils.collection_utils import group_by


class TestGroupBy(unittest.TestCase):
    def test_nested_key_grouping(self):
        sample = [
            {"score": 10, "metadata": {"url": "a"}},
            {"score": 20, "metadata": {"url": "b"}},
            {"score": 10, "metadata": {"url": "a"}},
        ]
        expected = [
            {
                "group": "a",
                "items": [
                    {"score": 10, "metadata": {"url": "a"}},
                    {"score": 10, "metadata": {"url": "a"}},
                ],
            },
            {
                "group": "b",
                "items": [
                    {"score": 20, "metadata": {"url": "b"}},
                ],
            },
        ]
        result = group_by(sample, "metadata['url']")
        self.assertEqual(sorted(result, key=lambda x: x["group"]), sorted(
            expected, key=lambda x: x["group"]))

    def test_class_object_grouping(self):
        class Meta:
            def __init__(self, url):
                self.url = url

        class Item:
            def __init__(self, score, metadata):
                self.score = score
                self.metadata = metadata

        sample = [
            Item(10, Meta("a")),
            Item(10, Meta("a")),
            Item(20, Meta("b"))
        ]

        result = group_by(sample, "metadata['url']")
        groups = {g["group"]: len(g["items"]) for g in result}
        expected = {"a": 2, "b": 1}
        self.assertEqual(groups, expected)


if __name__ == '__main__':
    unittest.main()
