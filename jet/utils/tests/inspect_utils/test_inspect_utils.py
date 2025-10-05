import pytest
import os
from unittest.mock import patch
from jet.utils.inspect_utils import (
    validate_filepath,
    truncate_value,
    log_filtered_stack_trace,
    inspect_original_script_path,
    get_stack_frames,
    find_stack_frames,
    get_current_running_function,
    get_method_info,
)

@pytest.fixture
def mock_inspect_stack():
    with patch("inspect.stack") as mock_stack:
        yield mock_stack

@pytest.fixture
def mock_sys_modules():
    with patch("sys.modules") as mock_modules:
        yield mock_modules

@pytest.fixture
def mock_logger():
    with patch("jet.utils.inspect_utils.logger") as mock_log:
        yield mock_log

class TestValidateFilepath:
    def test_valid_filepath(self):
        # Given: A filepath that includes a valid path
        filepath = "/path/to/Jet_Projects/script.py"
        expected = True

        # When: Validating the filepath
        result = validate_filepath(filepath)

        # Then: The filepath should be valid
        assert result == expected

    def test_invalid_filepath_excluded(self):
        # Given: A filepath that includes an excluded path
        filepath = "/path/to/site-packages/script.py"
        expected = False

        # When: Validating the filepath
        result = validate_filepath(filepath)

        # Then: The filepath should be invalid
        assert result == expected

    def test_invalid_filepath_no_included_path(self):
        # Given: A filepath that does not include a valid path
        filepath = "/path/to/other/script.py"
        expected = False

        # When: Validating the filepath
        result = validate_filepath(filepath)

        # Then: The filepath should be invalid
        assert result == expected

class TestTruncateValue:
    def test_truncate_long_string(self):
        # Given: A long string exceeding MAX_LOG_LENGTH
        input_value = "a" * 150
        expected = "a" * 100 + "..."

        # When: Truncating the value
        result = truncate_value(input_value)

        # Then: The string should be truncated
        assert result == expected

    def test_truncate_list(self):
        # Given: A list with more than 2 items
        input_value = [1, 2, 3, 4]
        expected = [1, 2, "..."]

        # When: Truncating the list
        result = truncate_value(input_value)

        # Then: The list should be truncated
        assert result == expected

    def test_truncate_dict(self):
        # Given: A dictionary with more than 5 items
        input_value = {f"key{i}": f"value{i}" for i in range(6)}
        expected = {f"key{i}": f"value{i}" for i in range(5)}

        # When: Truncating the dictionary
        result = truncate_value(input_value)

        # Then: The dictionary should be truncated
        assert result == expected

class TestLogFilteredStackTrace:
    def test_log_filtered_stack_trace(self, mock_logger):
        # Given: An exception with a stack trace
        try:
            raise ValueError("Test error")
        except ValueError as e:
            exc = e
        tb_frame = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script.py',
            'lineno': 10,
            'name': 'test_function',
            'line': 'raise ValueError("Test error")'
        })
        with patch("traceback.extract_tb", return_value=[tb_frame]):

            # When: Logging the filtered stack trace
            log_filtered_stack_trace(exc)

            # Then: Logger should be called with correct messages
            mock_logger.newline.assert_called_once()
            mock_logger.warning.assert_any_call(
                "Stack [0]: File \"/path/to/Jet_Projects/script.py\", line 10, in test_function"
            )
            mock_logger.error.assert_any_call("raise ValueError(\"Test error\")")

class TestInspectOriginalScriptPath:
    def test_valid_stack_frames(self, mock_inspect_stack):
        # Given: A stack with valid frames
        frame1 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script.py',
            'function': 'main',
            'lineno': 10,
            'code_context': ['main()'],
            'index': 0
        })
        frame2 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/other.py',
            'function': 'func',
            'lineno': 20,
            'code_context': ['func()'],
            'index': 1
        })
        mock_inspect_stack.return_value = [frame2, frame1]
        expected = {
            "first": {
                "filepath": os.path.abspath('/path/to/Jet_Projects/script.py'),
                "filename": "script.py",
                "function": "main",
                "lineno": 10,
                "code_context": ['main()'],
            },
            "last": {
                "filepath": os.path.abspath('/path/to/Jet_Projects/other.py'),
                "filename": "other.py",
                "function": "func",
                "lineno": 20,
                "code_context": ['func()'],
            }
        }

        # When: Inspecting the original script path
        result = inspect_original_script_path()

        # Then: The result should match the expected structure
        assert result == expected

    def test_no_valid_frames(self, mock_inspect_stack):
        # Given: A stack with no valid frames
        frame = type('Frame', (), {
            'filename': '/path/to/site-packages/script.py',
            'function': 'main',
            'lineno': 10,
            'code_context': ['main()'],
            'index': 0
        })
        mock_inspect_stack.return_value = [frame]
        expected = None

        # When: Inspecting the original script path
        result = inspect_original_script_path()

        # Then: The result should be None
        assert result == expected

class TestGetStackFrames:
    def test_get_stack_frames_limited(self, mock_inspect_stack):
        # Given: A stack with multiple frames and a max_frames limit
        frame1 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script1.py',
            'lineno': 10,
            'function': 'func1',
            'code_context': ['func1()'],
            'index': 0
        })
        frame2 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script2.py',
            'lineno': 20,
            'function': 'func2',
            'code_context': ['func2()'],
            'index': 1
        })
        mock_inspect_stack.return_value = [frame1, frame2]
        expected = [
            {
                'index': 1,
                'filename': '/path/to/Jet_Projects/script2.py',
                'lineno': 20,
                'function': 'func2',
                'code_context': ['func2()']
            }
        ]

        # When: Getting stack frames with max_frames=1
        result = get_stack_frames(max_frames=1)

        # Then: Only the last frame should be returned
        assert result == expected

class TestFindStackFrames:
    def test_find_stack_frames_with_text(self, mock_inspect_stack):
        # Given: A stack with frames containing specific text
        frame1 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script1.py',
            'lineno': 10,
            'function': 'func1',
            'code_context': ['print("test")'],
            'index': 0
        })
        frame2 = type('Frame', (), {
            'filename': '/path/to/Jet_Projects/script2.py',
            'lineno': 20,
            'function': 'func2',
            'code_context': ['other_code'],
            'index': 1
        })
        mock_inspect_stack.return_value = [frame1, frame2]
        expected = [
            {
                'index': 0,
                'filename': '/path/to/Jet_Projects/script1.py',
                'lineno': 10,
                'function': 'func1',
                'code_context': ['print("test")']
            }
        ]

        # When: Finding stack frames with text "test"
        result = find_stack_frames("test")

        # Then: Only frames containing "test" should be returned
        assert result == expected

class TestGetCurrentRunningFunction:
    def test_get_current_running_function(self, mock_inspect_stack):
        # Given: A stack with a current function
        frame = type('Frame', (), {
            'function': 'test_function',
            'filename': '/path/to/Jet_Projects/script.py',
            'lineno': 10,
            'code_context': ['test_function()'],
            'index': 1
        })
        mock_inspect_stack.return_value = [None, frame]
        expected = "test_function"

        # When: Getting the current running function
        result = get_current_running_function()

        # Then: The function name should be returned
        assert result == expected

class TestGetMethodInfo:
    def test_get_method_info(self):
        # Given: A sample class method with type hints and docstring
        class TestClass:
            def sample_method(self, name: str, age: int) -> bool:
                """Sample method description.

                Args:
                    name: The name of the person.
                    age: The age of the person.
                Returns:
                    bool: True if successful.
                """
                return True

        method = TestClass.sample_method
        expected = {
            "name": "sample_method",
            "description": "Sample method description.",
            "parameters": {
                "type": "object",
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string", "description": "The name of the person."},
                    "age": {"type": "integer", "description": "The age of the person."}
                }
            }
        }

        # When: Getting method info
        result = get_method_info(method)

        # Then: The method info should match the expected structure
        assert result == expected
