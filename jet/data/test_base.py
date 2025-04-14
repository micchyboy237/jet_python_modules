import unittest
from jet.data.base import BaseModel, create_dynamic_model, extract_titles_descriptions


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

        Model = create_dynamic_model(json_schema, nested=True)
        instance = Model(user={"id": "abc123", "roles": ["admin", "editor"]})

        self.assertEqual(instance.user.id, "abc123")
        self.assertIn("admin", instance.user.roles)

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

        Model = create_dynamic_model(json_schema, nested=True)
        sample = {
            "items": [
                {"label": "One", "amount": 1.5},
                {"label": "Two"}
            ]
        }

        instance = Model(**sample)
        self.assertEqual(instance.items[0].label, "One")

        schema = Model.model_json_schema()
        self.assertEqual(schema["properties"]["items"]["type"], "array")

    def test_model_json_schema_with_nested_object_list_recursive(self):
        json_schema = {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string"},
                            "value": {"type": "number"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["timestamp", "value"]
                    }
                }
            },
            "required": ["records"]
        }

        Model = create_dynamic_model(json_schema, nested=True)

        sample = {
            "records": [
                {
                    "timestamp": "2024-04-01T10:00:00Z",
                    "value": 123.45,
                    "tags": ["sensor-a", "room-1"]
                }
            ]
        }

        instance = Model(**sample)
        self.assertEqual(instance.records[0].timestamp, "2024-04-01T10:00:00Z")

        schema = Model.model_json_schema()

        def resolve_ref_chain(ref_obj):
            """Resolve nested $refs from definitions until the final object."""
            ref = ref_obj.get("$ref")
            if not ref:
                return ref_obj
            ref_key = ref.split("/")[-1]
            defs = schema.get("$defs") or schema.get("definitions", {})
            resolved = defs.get(ref_key, {})
            while "$ref" in resolved:
                ref = resolved["$ref"]
                ref_key = ref.split("/")[-1]
                resolved = defs.get(ref_key, {})
            return resolved

        records_schema = schema["properties"]["records"]
        items_schema = resolve_ref_chain(records_schema["items"])

        self.assertEqual(items_schema.get("type"), "object")

    def test_model_json_schema_with_nested_object_list_flat(self):
        json_schema = {
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "timestamp": {"type": "string"},
                            "value": {"type": "number"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["timestamp", "value"]
                    }
                }
            },
            "required": ["records"]
        }

        Model = create_dynamic_model(json_schema, nested=False)

        sample = {
            "records": [
                {
                    "timestamp": "2024-04-01T10:00:00Z",
                    "value": 123.45,
                    "tags": ["sensor-a", "room-1"]
                }
            ]
        }

        instance = Model(**sample)
        self.assertEqual(instance.records[0]["tags"], ["sensor-a", "room-1"])

        schema = Model.model_json_schema()
        self.assertEqual(schema["properties"]["records"]["type"], "array")
        # No guarantee that properties are present in flat mode
        self.assertNotIn(
            "properties", schema["properties"]["records"]["items"])

    def test_custom_base_model_override(self):
        class CustomOverride(BaseModel):
            @classmethod
            def model_json_schema(cls, *args, **kwargs):
                schema = super().model_json_schema(*args, **kwargs)
                schema["x-custom"] = "injected"
                return schema

        schema = {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"]
        }

        Model = create_dynamic_model(
            schema, model_name="IDModel", base_model=CustomOverride)

        model_schema = Model.model_json_schema()
        self.assertEqual(model_schema["x-custom"], "injected")

    def test_original_schema_is_preserved(self):
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "count": {"type": "integer"}
            },
            "required": ["id"]
        }
        expected = {
            "properties": {
                "id": {
                    "title": "Id",
                    "type": "string"
                },
                "count": {
                    "anyOf": [
                        {
                            "type": "integer"
                        },
                        {
                            "type": "null"
                        }
                    ],
                    "default": None,
                    "title": "Count"
                }
            },
            "required": [
                "id"
            ],
            "title": "DynamicModel",
            "type": "object"
        }

        Model = create_dynamic_model(schema)
        self.assertEqual(Model.model_json_schema(), expected)


class TestExtractTitlesDescriptions(unittest.TestCase):
    def test_basic_schema(self):
        schema = {
            "title": "Person",
            "description": "A person object",
            "type": "object"
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Person", "description": "A person object"}
        ]
        self.assertEqual(result, expected)

    def test_nested_properties(self):
        schema = {
            "type": "object",
            "title": "Person",
            "description": "A person object",
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name",
                    "description": "The person's name"
                },
                "address": {
                    "type": "object",
                    "title": "Address",
                    "description": "The person's address",
                    "properties": {
                        "street": {
                            "type": "string",
                            "title": "Street",
                            "description": "Street name"
                        }
                    }
                }
            }
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Person", "description": "A person object"},
            {"path": "name", "title": "Name", "description": "The person's name"},
            {"path": "address", "title": "Address",
                "description": "The person's address"},
            {"path": "address.street", "title": "Street",
                "description": "Street name"}
        ]
        self.assertEqual(result, expected)

    def test_array_items(self):
        schema = {
            "type": "array",
            "title": "Hobbies",
            "description": "List of hobbies",
            "items": {
                "type": "string",
                "title": "Hobby",
                "description": "A single hobby"
            }
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Hobbies", "description": "List of hobbies"},
            {"path": "items", "title": "Hobby", "description": "A single hobby"}
        ]
        self.assertEqual(result, expected)

    def test_allOf_combiner(self):
        schema = {
            "type": "object",
            "title": "Person",
            "description": "A person object",
            "allOf": [
                {
                    "type": "object",
                    "title": "NamePart",
                    "description": "Name details",
                    "properties": {
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "The person's name"
                        }
                    }
                }
            ]
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Person", "description": "A person object"},
            {"path": "allOf[0]", "title": "NamePart",
                "description": "Name details"},
            {"path": "allOf[0].name", "title": "Name",
                "description": "The person's name"}
        ]
        self.assertEqual(result, expected)

    def test_missing_title_or_description(self):
        schema = {
            "type": "object",
            "title": "Person",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The person's name"
                },
                "age": {
                    "type": "integer",
                    "title": "Age"
                }
            }
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Person", "description": None},
            {"path": "name", "title": None, "description": "The person's name"},
            {"path": "age", "title": "Age", "description": None}
        ]
        self.assertEqual(result, expected)

    def test_empty_schema(self):
        schema = {}
        result = extract_titles_descriptions(schema)
        expected = []
        self.assertEqual(result, expected)

    def test_additional_properties(self):
        schema = {
            "type": "object",
            "title": "Config",
            "description": "Configuration object",
            "additionalProperties": {
                "type": "string",
                "title": "ConfigValue",
                "description": "A configuration value"
            }
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Config", "description": "Configuration object"},
            {"path": "additionalProperties", "title": "ConfigValue",
                "description": "A configuration value"}
        ]
        self.assertEqual(result, expected)

    def test_no_titles_or_descriptions(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
        result = extract_titles_descriptions(schema)
        expected = []
        self.assertEqual(result, expected)

    def test_nested_empty_objects(self):
        schema = {
            "type": "object",
            "title": "Root",
            "description": "Root object",
            "properties": {
                "child": {
                    "type": "object",
                    "properties": {
                        "grandchild": {
                            "type": "object"
                        }
                    }
                }
            }
        }
        result = extract_titles_descriptions(schema)
        expected = [
            {"path": "", "title": "Root", "description": "Root object"}
        ]
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
