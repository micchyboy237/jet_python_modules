import unittest
from typing import Optional
from pydantic import BaseModel, ValidationError
from jet.validation import pydantic_validate_json


class UserModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


class TestPydanticModelValidation(unittest.TestCase):

    def test_valid_model(self):
        valid_data = {
            'name': 'John Doe',
            'age': 30,
            'email': 'johndoe@example.com'
        }
        user = UserModel(**valid_data)
        response = pydantic_validate_json(user)
        self.assertTrue(response['is_valid'])
        self.assertEqual(response['data'], valid_data)
        self.assertIsNone(response['errors'])

    def test_invalid_model_missing_field(self):
        valid_data = {
            'name': 'John Doe',
            'age': 30
        }
        user = UserModel(**valid_data)
        response = pydantic_validate_json(user)
        self.assertTrue(response['is_valid'])
        self.assertIn('email: field required', response['errors'])

    def test_invalid_model_wrong_type(self):
        invalid_data = {
            'name': 'John Doe',
            'age': '30'
        }
        user = UserModel(**invalid_data)
        response = pydantic_validate_json(user)
        self.assertFalse(response['is_valid'])
        self.assertIn('age: value is not a valid integer', response['errors'])

    def test_invalid_json(self):
        invalid_data = {
            'name': 'John Doe',
            'age': 'thirty'  # Invalid age
        }
        with self.assertRaises(ValidationError):
            user = UserModel(**invalid_data)  # Should raise a validation error
            pydantic_validate_json(user)


if __name__ == '__main__':
    unittest.main()
