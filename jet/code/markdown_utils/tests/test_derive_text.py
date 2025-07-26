import pytest
from typing import Dict, Any
from jet.code.markdown_utils._markdown_parser import derive_text


class TestDeriveText:
    def test_header_token(self):
        # Given: A header token with level and content
        token: Dict[str, Any] = {
            'type': 'header',
            'level': 2,
            'content': '## My Header'
        }
        expected = '## My Header'

        # When: Deriving text from the header token
        result = derive_text(token)

        # Then: The result should match the expected content
        assert result == expected

    def test_unordered_list_without_prefix(self):
        # Given: An unordered list token with items that have no prefix
        token: Dict[str, Any] = {
            'type': 'unordered_list',
            'meta': {
                'items': [
                    {'text': 'Item one', 'task_item': False},
                    {'text': 'Item two', 'task_item': False}
                ]
            }
        }
        expected = '- Item one\n- Item two'

        # When: Deriving text from the unordered list token
        result = derive_text(token)

        # Then: The result should prepend dash prefix to each item
        assert result == expected

    def test_unordered_list_with_existing_prefix(self):
        # Given: An unordered list token with items that already have a prefix
        token: Dict[str, Any] = {
            'type': 'unordered_list',
            'meta': {
                'items': [
                    {'text': '- Item one', 'task_item': False},
                    {'text': '* Item two', 'task_item': False}
                ]
            }
        }
        expected = '- Item one\n* Item two'

        # When: Deriving text from the unordered list token
        result = derive_text(token)

        # Then: The result should keep existing prefixes
        assert result == expected

    def test_ordered_list_without_prefix(self):
        # Given: An ordered list token with items that have no prefix
        token: Dict[str, Any] = {
            'type': 'ordered_list',
            'meta': {
                'items': [
                    {'text': 'First item', 'task_item': False},
                    {'text': 'Second item', 'task_item': False}
                ]
            }
        }
        expected = '1. First item\n2. Second item'

        # When: Deriving text from the ordered list token
        result = derive_text(token)

        # Then: The result should prepend numbered prefixes
        assert result == expected

    def test_ordered_list_with_existing_prefix(self):
        # Given: An ordered list token with items that already have a prefix
        token: Dict[str, Any] = {
            'type': 'ordered_list',
            'meta': {
                'items': [
                    {'text': '1. First item', 'task_item': False},
                    {'text': 'a) Second item', 'task_item': False}
                ]
            }
        }
        expected = '1. First item\na) Second item'

        # When: Deriving text from the ordered list token
        result = derive_text(token)

        # Then: The result should keep existing prefixes
        assert result == expected

    def test_unordered_list_with_checkboxes(self):
        # Given: An unordered list token with items that have checkboxes
        token: Dict[str, Any] = {
            'type': 'unordered_list',
            'meta': {
                'items': [
                    {'text': 'Task one', 'task_item': True, 'checked': True},
                    {'text': 'Task two', 'task_item': True, 'checked': False}
                ]
            }
        }
        expected = '- [x] Task one\n- [ ] Task two'

        # When: Deriving text from the unordered list token
        result = derive_text(token)

        # Then: The result should include checkboxes with dash prefix
        assert result == expected

    def test_table_token(self):
        # Given: A table token with header and rows
        token: Dict[str, Any] = {
            'type': 'table',
            'meta': {
                'header': ['Name', 'Age'],
                'rows': [['Alice', '30'], ['Bob', '25']]
            }
        }
        expected = '| Name  | Age |\n| ----- | --- |\n| Alice | 30  |\n| Bob   | 25  |'

        # When: Deriving text from the table token
        result = derive_text(token)

        # Then: The result should format the table correctly
        assert result == expected

    def test_code_token(self):
        # Given: A code token with content
        token: Dict[str, Any] = {
            'type': 'code',
            'content': '```python\nprint("Hello")\n```'
        }
        expected = 'print("Hello")'

        # When: Deriving text from the code token
        result = derive_text(token)

        # Then: The result should strip code block delimiters
        assert result == expected

    def test_paragraph_token(self):
        # Given: A paragraph token with content
        token: Dict[str, Any] = {
            'type': 'paragraph',
            'content': 'This is a paragraph.'
        }
        expected = 'This is a paragraph.'

        # When: Deriving text from the paragraph token
        result = derive_text(token)

        # Then: The result should match the content
        assert result == expected
