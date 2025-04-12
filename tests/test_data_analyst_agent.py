from unittest.mock import MagicMock

import plotly.graph_objects as go

from app.tools.data_analyst_agent import DataVisualizationAgent


def test_init_default_prompt(mock_llm):
    agent = DataVisualizationAgent(model=mock_llm)
    assert agent.system_prompt_template is not None


def test_init_custom_prompt(mock_llm):
    custom_prompt = "Custom prompt"
    agent = DataVisualizationAgent(model=mock_llm, system_prompt_template=custom_prompt)
    assert agent.system_prompt_template == custom_prompt


def test_setup_logging(temp_dir):
    agent = DataVisualizationAgent(model=MagicMock(), log=True, log_path=str(temp_dir))
    assert agent.logger is not None


def test_format_data_summary(mock_df):
    agent = DataVisualizationAgent(model=MagicMock())
    summary = agent._format_data_summary(mock_df)
    assert "DataFrame Shape: 3 rows, 2 columns" in summary
    assert "A: int64" in summary


def test_generate_visualization_success(mock_llm, mock_df):
    agent = DataVisualizationAgent(model=mock_llm)
    mock_llm.invoke.return_value = MagicMock(
        content='```json\n{"code": "fig = go.Figure()", "explanation": "test"}\n```'
    )
    response = agent.generate_visualization(mock_df, "create a chart")
    assert response["success"] is True


def test_generate_visualization_json_error(mock_llm, mock_df):
    agent = DataVisualizationAgent(model=mock_llm)
    mock_llm.invoke.return_value = MagicMock(content="```python\nfig = go.Figure()\nfig\n```")
    response = agent.generate_visualization(mock_df, "create a chart")
    assert response["success"] is True


def test_execute_visualization_code_error(mock_llm, mock_df):
    agent = DataVisualizationAgent(model=mock_llm)
    agent.visualization_code = "invalid code"
    response = agent._execute_visualization_code(mock_df, max_retries=1)
    assert response["success"] is False
    assert "error" in response


def test_request_code_fix(mock_llm, mock_df):
    agent = DataVisualizationAgent(model=mock_llm)
    agent.visualization_code = "broken code"
    mock_llm.invoke.return_value = MagicMock(content="```python\nfig = go.Figure()\nfig\n```")
    agent._request_code_fix(mock_df, "syntax error")
    assert agent.visualization_code == "fig = go.Figure()\nfig"


def test_getters(mock_llm):
    agent = DataVisualizationAgent(model=mock_llm)
    agent.response = {"test": "value"}
    agent.visualization_code = "test code"
    agent.plotly_figure = go.Figure()
    assert agent.get_response() == {"test": "value"}
    assert agent.get_visualization_code() == "test code"
    assert agent.get_visualization_code(format_markdown=True) == "```python\ntest code\n```"
    assert isinstance(agent.get_plotly_figure(), go.Figure)
