import os
import pytest
from jet.logger import CustomLogger


@pytest.fixture
def logger(tmp_path):
    """
    Fixture to provide a CustomLogger instance for tests.
    Given: A temporary directory for test logs
    When: The logger fixture is used in a test
    Then: It provides a CustomLogger writing to a test-specific log file
    """
    log_file = os.path.join(tmp_path, "test.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Initialized logger for test with log file: {log_file}")

    yield logger
    # No close() call, as CustomLogger likely manages file handles internally
