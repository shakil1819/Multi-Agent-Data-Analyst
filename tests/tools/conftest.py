import os
import sys

from unittest.mock import patch

import pytest


# Add the parent directory to sys.path to allow imports from the app package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# Mock the to_sql method for pandas DataFrames
@pytest.fixture(autouse=True)
def mock_pandas_to_sql() -> None:
    with patch("pandas.DataFrame.to_sql") as mock:
        yield mock
