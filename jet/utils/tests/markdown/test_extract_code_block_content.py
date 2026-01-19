

from jet.utils.markdown import extract_code_block_content


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
