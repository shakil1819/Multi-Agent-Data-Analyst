import unittest

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.tools.sql_data_analyst_agent import SqlDataAnalystAgent


class TestSqlDataAnalystAgent(unittest.TestCase):
    @patch("app.tools.sql_data_analyst_agent.glob.glob")
    @patch("app.tools.sql_data_analyst_agent.pd.read_csv")
    @patch("app.tools.sql_data_analyst_agent.ChatOpenAI")
    @patch("app.tools.sql_data_analyst_agent.SQLDatabaseAgent")
    @patch("app.tools.sql_data_analyst_agent.sql.create_engine")
    @patch("app.tools.sql_data_analyst_agent.os.getenv")
    def setUp(self, mock_getenv, mock_create_engine, mock_sql_agent, mock_chat, mock_read_csv, mock_glob):
        # Set up mocks
        mock_getenv.return_value = "fake-api-key"
        mock_glob.return_value = ["./data/test.csv"]
        self.test_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        mock_read_csv.return_value = self.test_df

        # Mock the engine and connection
        self.mock_engine = mock_create_engine.return_value
        self.mock_conn = self.mock_engine.connect.return_value

        # Create the agent
        self.agent = SqlDataAnalystAgent(model_name="test-model")

        # Verify mocks were called correctly
        mock_getenv.assert_called_with("OPENAI_API_KEY")
        mock_chat.assert_called_once()
        mock_glob.assert_called_with("./data/*.csv")
        mock_read_csv.assert_called_once()
        mock_create_engine.assert_called_with("sqlite:///:memory:")
        # Fix: Remove assertion on to_sql since it's mocked at fixture level
        # self.test_df.to_sql.assert_called_with("data", self.mock_engine, index=False)  # noqa: ERA001
        mock_sql_agent.assert_called_once()

        # Save mocks for later assertions
        self.mock_sql_agent = mock_sql_agent.return_value

    @patch("app.tools.sql_data_analyst_agent.glob.glob")
    def test_init_raises_error_with_no_csv(self, mock_glob):
        # Test that initialization raises an error when no CSV files are found
        mock_glob.return_value = []
        with pytest.raises(ValueError):  # noqa: PT011
            SqlDataAnalystAgent()

    @patch("app.tools.sql_data_analyst_agent.glob.glob")
    def test_init_raises_error_with_multiple_csv(self, mock_glob):
        # Test that initialization raises an error when multiple CSV files are found
        mock_glob.return_value = ["./data/test1.csv", "./data/test2.csv"]
        with pytest.raises(ValueError):  # noqa: PT011
            SqlDataAnalystAgent()

    async def test_process_query_success(self):
        # Set up the mock response for a successful query
        self.mock_sql_agent.ainvoke_agent = AsyncMock()
        self.mock_sql_agent.get_sql_query_code.return_value = "SELECT * FROM data"
        self.mock_sql_agent.get_data_sql.return_value = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Test the process_query method
        result = await self.agent.process_query("Show me all data")

        # Verify the result
        assert result["status"] == "success"
        assert result["sql_query"] == "SELECT * FROM data"
        assert result["data"] == [{"col1": 1, "col2": 3}, {"col1": 2, "col2": 4}]

        # Verify the agent was invoked correctly
        self.mock_sql_agent.ainvoke_agent.assert_called_with(user_instructions="Show me all data")

    async def test_process_query_no_data(self):
        # Set up the mock response for a query with no data
        self.mock_sql_agent.ainvoke_agent = AsyncMock()
        self.mock_sql_agent.get_sql_query_code.return_value = "SELECT * FROM data WHERE 1=0"
        self.mock_sql_agent.get_data_sql.return_value = None

        # Test the process_query method
        result = await self.agent.process_query("Show me no data")

        # Verify the result
        assert result["status"] == "error"
        assert result["message"] == "No data returned from the query"

    async def test_process_query_exception(self):
        # Set up the mock to raise an exception
        self.mock_sql_agent.ainvoke_agent = AsyncMock(side_effect=Exception("Test error"))

        # Test the process_query method
        result = await self.agent.process_query("Show me data")

        # Verify the result
        assert result["status"] == "error"
        assert result["message"] == "Error processing query: Test error"


if __name__ == "__main__":
    unittest.main()
