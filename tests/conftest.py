from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import sqlalchemy as sql

from langchain_openai import ChatOpenAI
from pydantic_ai.usage import UsageLimits


# Fixture for mocking the LLM
@pytest.fixture
def mock_llm():
    return MagicMock(spec=ChatOpenAI)


@pytest.fixture
def mock_db_connection():
    connection = MagicMock(spec=sql.engine.base.Connection)
    connection.engine = MagicMock(spec=sql.engine.Engine)
    return connection


@pytest.fixture(autouse=True)
def patch_sql_inspect():
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = ["table"]
    mock_inspector.get_columns.return_value = [{"name": "x", "type": "INTEGER"}]
    with patch("sqlalchemy.inspect", return_value=mock_inspector):
        yield


# Fixture for a sample DataFrame
@pytest.fixture
def mock_df():
    return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})


# Fixture for usage limits
@pytest.fixture
def usage_limits():
    return UsageLimits(total_tokens_limit=4000, request_limit=10)


# Fixture for mocking OrchestratorAgent
@pytest.fixture
def mock_orchestrator_agent():
    with patch("app.agent_orchestrator.orchestrator_agent") as mock:
        yield mock


# Fixture for mocking SQLDataAnalysisAgent
@pytest.fixture
def mock_sql_agent():
    with patch("app.tools.sql_data_analyst_agent.SQLDataAnalysisAgent") as mock:
        yield mock


# Fixture for mocking DataVisualizationAgent
@pytest.fixture
def mock_visualization_agent():
    with patch("app.tools.data_analyst_agent.DataVisualizationAgent") as mock:
        yield mock


# Fixture for temporary directory
@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path
