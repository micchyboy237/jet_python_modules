import unittest
from jsonschema.exceptions import ValidationError
from jet.validation import schema_validate_json, ValidationResponse


class TestSchemaValidateJson(unittest.TestCase):
    def setUp(self):
        # Example schema for testing
        self.resume_schema = {
            "type": "object",
            "properties": {
                "scope_of_work": {"type": "array", "items": {"type": "string"}},
                "job_description": {"type": "string"},
                "qualifications": {"type": "array", "items": {"type": "string"}},
                "responsibilities": {"type": "array", "items": {"type": "string"}},
                "tech_stack": {
                    "type": "object",
                    "properties": {
                        "frontend": {"type": "array", "items": {"type": "string"}},
                        "backend": {"type": "array", "items": {"type": "string"}},
                        "database": {"type": "array", "items": {"type": "string"}},
                        "other": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["frontend", "backend", "database", "other"],
                },
                "level": {"type": "string", "enum": ["entry", "mid", "senior"]},
                "salary": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "integer"},
                        "max": {"type": "integer"},
                        "currency": {"type": "string"},
                        "period": {"type": "string"},
                    },
                    "required": ["min", "max", "currency", "period"],
                },
            },
            "required": [
                "scope_of_work",
                "job_description",
                "qualifications",
                "responsibilities",
                "tech_stack",
                "level",
                "salary",
            ],
        }

    def test_invalid_json_sample(self):
        invalid_json_sample = """{
            "scope_of_work": ["backend", "web"],
            "job_description": "Develop and maintain web applications.",
            "qualifications": ["5+ years experience", "Proficient in Python"],
            "responsibilities": ["Web development", "Backend development"],
            "tech_stack": {
                "frontend": ["React", "Vue.js"],
                "backend": ["Node.js", "Python"],
                "database": ["PostgreSQL", "MongoDB"],
                "other": ["AWS"]
            },
            "level": "expert",
            "salary": {
                "min": 900,
                "max": 1100,
                "currency": "USD",
                "period": "monthly"
            }
        }"""
        response = schema_validate_json(
            invalid_json_sample, self.resume_schema)
        self.assertFalse(response["is_valid"])
        self.assertIn("level", response["errors"][0])

    def test_valid_json_incorrect_sample(self):
        valid_json_incorrect_sample = """{
            "scope_of_work": ["backen", "web"],
            "job_description": "Develop and maintain web applications.",
            "qualifications": ["5+ years experience", "Proficient in Python"],
            "responsibilities": ["Web development", "Backend development"],
            "tech_stack": {
                "frontend": ["React", "Vue.js"],
                "backend": ["Node.js", "Python"],
                "database": ["PostgreSQL", "MongoDB"],
                "other": ["AWS"]
            },
            "level": "Mid-Senior level",
            "salary": {
                "min": 900,
                "max": "1100",
                "currency": "Php",
                "period": "Month"
            }
        }"""
        response = schema_validate_json(
            valid_json_incorrect_sample, self.resume_schema)
        self.assertFalse(response["is_valid"])
        self.assertTrue(any("level" in error for error in response["errors"]))

    def test_valid_json_correct_sample(self):
        valid_json_correct_sample = """{
            "scope_of_work": ["backend", "web"],
            "job_description": "Develop and maintain web applications.",
            "qualifications": ["5+ years experience", "Proficient in Python"],
            "responsibilities": ["Web development", "Backend development"],
            "tech_stack": {
                "frontend": ["React", "Vue.js"],
                "backend": ["Node.js", "Python"],
                "database": ["PostgreSQL", "MongoDB"],
                "other": ["AWS"]
            },
            "level": "mid",
            "salary": {
                "min": 900,
                "max": 1100,
                "currency": "USD",
                "period": "monthly"
            }
        }"""
        response = schema_validate_json(
            valid_json_correct_sample, self.resume_schema)
        self.assertTrue(response["is_valid"])
        self.assertIsNotNone(response["data"])
        self.assertIsNone(response["errors"])


if __name__ == "__main__":
    unittest.main()
