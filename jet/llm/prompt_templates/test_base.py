import unittest
from jet.llm.prompt_templates.base import create_dynamic_model


class TestDynamicModelCreation(unittest.TestCase):

    def test_simple_schema(self):
        json_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "is_active": {"type": "boolean"}
            },
            "required": ["name", "age"]
        }

        Model = create_dynamic_model(json_schema)
        instance = Model(name="Jet", age=35, is_active=True)

        self.assertEqual(instance.name, "Jet")
        self.assertEqual(instance.age, 35)
        self.assertTrue(instance.is_active)

        schema = Model.model_json_schema()
        self.assertIn("name", schema["properties"])
        self.assertEqual(schema["properties"]["age"]["type"], "integer")

    def test_optional_fields_and_nullables(self):
        json_schema = {
            "type": "object",
            "properties": {
                "nickname": {"type": ["string", "null"]},
                "score": {"type": ["number", "null"]}
            }
        }

        Model = create_dynamic_model(json_schema)
        instance = Model(nickname=None, score=99.5)

        self.assertIsNone(instance.nickname)
        self.assertEqual(instance.score, 99.5)

        schema = Model.model_json_schema()
        nickname_schema = schema["properties"]["nickname"]
        score_schema = schema["properties"]["score"]

        # Pydantic uses anyOf for unions like ["string", "null"]
        self.assertIn("anyOf", nickname_schema)
        types = {option["type"] for option in nickname_schema["anyOf"]}
        self.assertEqual(types, {"string", "null"})

        self.assertIn("anyOf", score_schema)
        types = {option["type"] for option in score_schema["anyOf"]}
        self.assertEqual(types, {"number", "null"})

    def test_nested_object(self):
        json_schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "roles": {
                            "type": "array",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["id"]
                }
            },
            "required": ["user"]
        }

        Model = create_dynamic_model(json_schema)
        instance = Model(user={"id": "abc123", "roles": ["admin", "editor"]})

        self.assertEqual(instance.user["id"], "abc123")
        self.assertIn("admin", instance.user["roles"])

        schema = Model.model_json_schema()
        self.assertIn("user", schema["properties"])

    def test_union_type_multiple_non_null(self):
        json_schema = {
            "type": "object",
            "properties": {
                "value": {"type": ["string", "integer"]},
            }
        }

        Model = create_dynamic_model(json_schema)
        instance = Model(value="hello")
        self.assertEqual(instance.value, "hello")

        instance = Model(value=42)
        self.assertEqual(instance.value, 42)

    def test_array_of_objects(self):
        json_schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "amount": {"type": "number"}
                        },
                        "required": ["label"]
                    }
                }
            },
            "required": ["items"]
        }

        Model = create_dynamic_model(json_schema)
        sample = {
            "items": [
                {"label": "One", "amount": 1.5},
                {"label": "Two"}
            ]
        }

        instance = Model(**sample)
        self.assertEqual(instance.items[0]["label"], "One")

        schema = Model.model_json_schema()
        self.assertEqual(schema["properties"]["items"]["type"], "array")


if __name__ == "__main__":
    unittest.main()
