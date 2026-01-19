import json
import pytest
import yaml
from config_utils import (
    extract_code_block_content,
    yaml_to_dict,
    yaml_to_json,
    yaml_file_to_json_file,
)


class TestExtractCodeBlockContent:
    """Tests for extract_code_block_content function"""

    def test_plain_text_without_code_block(self):
        # Given
        text = "Hello world\nThis is normal text"
        expected = "Hello world\nThis is normal text"

        # When
        result = extract_code_block_content(text)

        # Then
        assert result == expected

    def test_code_block_without_language(self):
        # Given
        text = """Some text before
```
print("hello")
return 42
```
Some text after"""

        expected = """print("hello")
return 42"""

        # When
        result = extract_code_block_content(text)

        # Then
        assert result == expected

    def test_code_block_with_specific_language(self):
        # Given
        text = """Explanation
```python
def hello():
    print("world")
```
More text"""

        expected = """def hello():
    print("world")"""

        # When
        result = extract_code_block_content(text, lang="python")

        # Then
        assert result == expected

    def test_multiple_code_blocks_returns_first_one(self):
        # Given
        text = """```js
console.log("first")
```

```python
print("second")
```"""

        expected = """console.log("first")"""

        # When
        result = extract_code_block_content(text)

        # Then
        assert result == expected

    def test_code_block_at_beginning(self):
        # Given
        text = """```yaml
key: value
list:
  - item1
  - item2
```"""

        expected = """key: value
list:
  - item1
  - item2"""

        # When
        result = extract_code_block_content(text, "yaml")

        # Then
        assert result == expected

    def test_no_closing_fence_returns_from_start_to_end(self):
        # Given
        text = """Some intro
```sql
SELECT * FROM users
WHERE active = true"""

        expected = """SELECT * FROM users
WHERE active = true"""

        # When
        result = extract_code_block_content(text, "sql")

        # Then
        assert result == expected

    def test_empty_code_block(self):
        # Given
        text = "```json\n\n```"
        expected = ""

        # When
        result = extract_code_block_content(text, "json")

        # Then
        assert result == expected


class TestYamlToDict:
    """Tests for yaml_to_dict function"""

    def test_valid_simple_yaml(self):
        # Given
        yaml_str = """
name: John
age: 30
active: true
skills:
  - python
  - yaml
"""
        expected = {
            "name": "John",
            "age": 30,
            "active": True,
            "skills": ["python", "yaml"],
        }

        # When
        result = yaml_to_dict(yaml_str)

        # Then
        assert result == expected

    def test_valid_yaml_with_quotes(self):
        # Given
        yaml_str = 'message: "Hello, world!"\npath: /api/v1/users'
        expected = {"message": "Hello, world!", "path": "/api/v1/users"}

        # When
        result = yaml_to_dict(yaml_str)

        # Then
        assert result == expected

    @pytest.mark.parametrize(
        "invalid_input,expected_error",
        [
            ("key: [unclosed list", yaml.YAMLError),
            ("!!python/object/apply:os.system ['calc.exe']", yaml.constructor.ConstructorError),
        ],
    )
    def test_unsafe_or_invalid_yaml_is_rejected(self, invalid_input, expected_error):
        # When/Then
        with pytest.raises(expected_error):
            yaml_to_dict(invalid_input)


class TestYamlToJson:
    """Tests for yaml_to_json function"""

    def test_conversion_with_default_indent(self):
        # Given
        yaml_str = """
person:
  name: Alice
  age: 28
"""
        expected_json = """{
  "person": {
    "name": "Alice",
    "age": 28
  }
}"""

        # When
        result = yaml_to_json(yaml_str)

        # Then
        assert result == expected_json

    def test_custom_indent(self):
        # Given
        yaml_str = "items: [apple, banana, cherry]"
        expected = "{\n    \"items\": [\n        \"apple\",\n        \"banana\",\n        \"cherry\"\n    ]\n}"

        # When
        result = yaml_to_json(yaml_str, indent=4)

        # Then
        assert result == expected

    def test_non_ascii_characters(self):
        # Given
        yaml_str = "greeting: สวัสดี\ncity: 서울"
        expected_contains = ["สวัสดี", "서울"]

        # When
        result = yaml_to_json(yaml_str)

        # Then
        assert all(word in result for word in expected_contains)


class TestYamlFileToJsonFile:
    """Tests for yaml_file_to_json_file function"""

    def test_file_conversion(self, tmp_path):
        # Given
        yaml_content = """
config:
  debug: true
  port: 8080
  hosts:
    - localhost
    - 127.0.0.1
"""

        input_file = tmp_path / "test_input.yaml"
        output_file = tmp_path / "test_output.json"

        input_file.write_text(yaml_content, encoding="utf-8")

        expected_data = {
            "config": {"debug": True, "port": 8080, "hosts": ["localhost", "127.0.0.1"]}
        }

        # When
        yaml_file_to_json_file(str(input_file), str(output_file))

        # Then
        assert output_file.exists()

        with open(output_file, encoding="utf-8") as f:
            result_data = json.load(f)

        assert result_data == expected_data