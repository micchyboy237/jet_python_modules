import os
import shutil
import pytest

from jet.scrapers.automation.clone_after_render import run_clone_after_render

TEST_URL = 'https://example.com'
OUTPUT_FOLDER = 'test_mirror_output'


class TestCloneAfterRender:
    def setup_method(self):
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)

    def teardown_method(self):
        if os.path.exists(OUTPUT_FOLDER):
            shutil.rmtree(OUTPUT_FOLDER)

    def test_run_clone_after_render(self):
        sample = TEST_URL
        expected = os.path.join(OUTPUT_FOLDER, 'index.html')

        run_clone_after_render(sample, OUTPUT_FOLDER)

        result = os.path.exists(expected)
        assert result is True
