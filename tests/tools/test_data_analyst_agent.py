import unittest

from unittest.mock import patch

import pandas as pd
import pytest

from app.tools.data_analyst_agent import DataAnalystAgent


class TestDataAnalystAgent(unittest.TestCase):
    @patch("app.tools.data_analyst_agent.glob.glob")
    @patch("app.tools.data_analyst_agent.pd.read_csv")
    @patch("app.tools.data_analyst_agent.ChatOpenAI")
    @patch("app.tools.data_analyst_agent.DataWranglingAgent")
    @patch("app.tools.data_analyst_agent.DataVisualizationAgent")
    @patch("app.tools.data_analyst_agent.PandasDataAnalyst")
    @patch("app.tools.data_analyst_agent.os.getenv")
    def setUp(
        self,
        mock_getenv,
        mock_pandas_analyst,
        mock_viz_agent,
        mock_wrangling_agent,
        mock_chat,
        mock_read_csv,
        mock_glob,
    ):
        # Set up mocks
        mock_getenv.return_value = "fake-api-key"
        mock_glob.return_value = ["./data/test.csv"]
        mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Create the agent
        self.agent = DataAnalystAgent(model_name="test-model")

        # Verify mocks were called correctly
        mock_getenv.assert_called_with("OPENAI_API_KEY")
        mock_chat.assert_called_once()
        mock_glob.assert_called_with("./data/*.csv")
        mock_read_csv.assert_called_once()
        mock_wrangling_agent.assert_called_once()
        mock_viz_agent.assert_called_once()
        mock_pandas_analyst.assert_called_once()

        # Save mocks for later assertions
        self.mock_pandas_analyst = mock_pandas_analyst.return_value

    @patch("app.tools.data_analyst_agent.glob.glob")
    def test_init_raises_error_with_no_csv(self, mock_glob):
        # Test that initialization raises an error when no CSV files are found
        mock_glob.return_value = []
        with pytest.raises(ValueError):  # noqa: PT011
            DataAnalystAgent()

    @patch("app.tools.data_analyst_agent.glob.glob")
    def test_init_raises_error_with_multiple_csv(self, mock_glob):
        # Test that initialization raises an error when multiple CSV files are found
        mock_glob.return_value = ["./data/test1.csv", "./data/test2.csv"]
        with pytest.raises(ValueError):  # noqa: PT011
            DataAnalystAgent()

    def test_process_query_chart_success(self):
        # Set up the mock response for a successful chart query
        self.mock_pandas_analyst.get_response.return_value = {
            "routing_preprocessor_decision": "chart",
            "plotly_error": False,
            "plotly_graph": '{"data": [{"x": [1, 2], "y": [3, 4], "type": "bar"}]}',
        }

        # Test the process_query method
        result = self.agent.process_query("Show me a chart of col1 vs col2")

        # Verify the result
        assert result["type"] == "chart"
        assert result["data"] == '{"data": [{"x": [1, 2], "y": [3, 4], "type": "bar"}]}'

        # Verify the agent was invoked correctly
        self.mock_pandas_analyst.invoke_agent.assert_called_with(
            user_instructions="Show me a chart of col1 vs col2",
            data_raw=self.agent.df,
        )

    def test_process_query_chart_no_data(self):
        # Set up the mock response for a chart query with no data
        self.mock_pandas_analyst.get_response.return_value = {
            "routing_preprocessor_decision": "chart",
            "plotly_error": False,
            "plotly_graph": None,
        }

        # Test the process_query method
        result = self.agent.process_query("Show me a chart of col1 vs col2")

        # Verify the result
        assert result["type"] == "error"
        assert result["message"] == "No valid chart data returned"

    def test_process_query_table_success(self):
        # Set up the mock response for a successful table query
        test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        self.mock_pandas_analyst.get_response.return_value = {
            "routing_preprocessor_decision": "table",
            "data_wrangled": test_df,
        }

        # Test the process_query method
        result = self.agent.process_query("Show me the data")

        # Verify the result
        assert result["type"] == "table"
        assert result["data"] == [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}]

    def test_process_query_exception(self):
        # Set up the mock to raise an exception
        self.mock_pandas_analyst.invoke_agent.side_effect = Exception("Test error")

        # Test the process_query method
        result = self.agent.process_query("Show me the data")

        # Verify the result
        assert result["type"] == "error"
        assert result["message"] == "Error processing query: Test error"


if __name__ == "__main__":
    unittest.main()
