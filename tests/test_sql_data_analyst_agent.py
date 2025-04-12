from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.tools.sql_data_analyst_agent import SQLDataAnalysisAgent


def test_init_with_connection(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    assert agent.connection == mock_db_connection


def test_init_with_engine_url(mock_llm):
    with patch("sqlalchemy.create_engine") as mock_create_engine:
        mock_engine = MagicMock()
        mock_connection = MagicMock()
        mock_create_engine.return_value = mock_engine
        mock_engine.connect.return_value = mock_connection
        agent = SQLDataAnalysisAgent(model=mock_llm, engine_url="sqlite:///:memory:")
        assert agent.connection == mock_connection


def test_init_missing_connection_and_engine_url(mock_llm):
    with pytest.raises(ValueError):  # noqa: PT011
        SQLDataAnalysisAgent(model=mock_llm)


def test_determine_analysis_type_visualization():
    agent = SQLDataAnalysisAgent(model=MagicMock(), connection=MagicMock())
    result = agent._determine_analysis_type("create a chart")
    assert result["needs_visualization"] is True


def test_determine_analysis_type_no_visualization():
    agent = SQLDataAnalysisAgent(model=MagicMock(), connection=MagicMock())
    result = agent._determine_analysis_type("select data")
    assert result["needs_visualization"] is False


def test_extract_code_from_response_markdown():
    agent = SQLDataAnalysisAgent(model=MagicMock(), connection=MagicMock())
    response = "```python\nprint('hello')\n```"
    code = agent._extract_code_from_response(response)
    assert code == "print('hello')"


def test_extract_code_from_response_plain():
    agent = SQLDataAnalysisAgent(model=MagicMock(), connection=MagicMock())
    response = "SELECT * FROM table"
    code = agent._extract_code_from_response(response)
    assert code == "SELECT * FROM table"


def test_generate_and_execute_sql_success(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    mock_llm.invoke.side_effect = [
        MagicMock(content="SELECT * FROM table"),
        MagicMock(
            content="def execute_sql_query(connection):\n    result = connection.execute(sql.text('SELECT * FROM table'))\n    return pd.DataFrame(result.fetchall(), columns=result.keys())"
        ),
    ]
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [(1, 2)]
    mock_result.keys.return_value = ["a", "b"]
    mock_db_connection.execute.return_value = mock_result
    agent._generate_and_execute_sql("get data", {"needs_visualization": False})
    assert isinstance(agent.get_data_sql(), pd.DataFrame)


def test_generate_and_execute_sql_error(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    mock_llm.invoke.side_effect = [
        MagicMock(content="SELECT * FROM table"),
        MagicMock(content="invalid code"),
    ]
    mock_db_connection.execute.side_effect = Exception("SQL error")
    agent._generate_and_execute_sql("get data", {"needs_visualization": False})
    assert "SQL execution failed" in agent.get_error()


def test_generate_visualization_success(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    agent._state["data_sql"] = pd.DataFrame({"x": [1, 2]})
    mock_llm.invoke.return_value = MagicMock(
        content="def create_visualization(df):\n    fig = go.Figure()\n    return fig"
    )
    agent._generate_visualization("create a chart")
    assert agent.get_plotly_graph() is not None


def test_generate_visualization_no_data(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    agent._generate_visualization("create a chart")
    assert agent.get_plotly_graph() is None


def test_reset_state(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    agent._state["error"] = "test error"
    agent._reset_state()
    assert agent._state["error"] is None


def test_get_database_schema(mock_db_connection):
    agent = SQLDataAnalysisAgent(model=MagicMock(), connection=mock_db_connection)
    mock_inspector = MagicMock()
    mock_inspector.get_table_names.return_value = ["table1"]
    mock_inspector.get_columns.return_value = [{"name": "col1", "type": "INT"}]
    with patch("sqlalchemy.inspect", return_value=mock_inspector):
        schema = agent._get_database_schema()
    assert "Table: table1" in schema
    assert "col1: INT" in schema


def test_getters(mock_llm, mock_db_connection):
    agent = SQLDataAnalysisAgent(model=mock_llm, connection=mock_db_connection)
    agent._state.update({"sql_query_code": "SELECT *", "error": "test error"})
    assert agent.get_sql_query_code() == "SELECT *"
    assert agent.get_sql_query_code(markdown=True) == "```sql\nSELECT *\n```"
    assert agent.get_error() == "test error"
