import glob
import os

import pandas as pd

from ai_data_science_team import (
    DataVisualizationAgent,
    DataWranglingAgent,
    PandasDataAnalyst,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv()


class DataAnalystAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the DataAnalystAgent with an OpenAI API key and model.

        Args:
            model_name (str): The OpenAI model to use (default: "gpt-4o-mini").

        Raises:
            ValueError: If exactly one CSV file is not found in the ./data directory.

        """
        # Initialize the language model
        api_key = os.getenv("OPENAI_API_KEY")
        # Convert string API key to SecretStr if not None
        secret_api_key = SecretStr(api_key) if api_key else None
        self.llm = ChatOpenAI(model=model_name, api_key=secret_api_key)

        # Load the dataset from the ./data directory
        csv_files = glob.glob("./data/*.csv")
        if len(csv_files) != 1:
            raise ValueError("Expected exactly one CSV file in ./data directory")
        self.df = pd.read_csv(csv_files[0])

        # Set up the data wrangling and visualization agents
        self.data_wrangling_agent = DataWranglingAgent(
            model=self.llm,
            log=False,
            bypass_recommended_steps=True,
            n_samples=100,
        )
        self.data_visualization_agent = DataVisualizationAgent(
            model=self.llm,
            n_samples=100,
            log=False,
        )

        # Initialize the PandasDataAnalyst with the configured agents
        self.pandas_data_analyst = PandasDataAnalyst(
            model=self.llm,
            data_wrangling_agent=self.data_wrangling_agent,
            data_visualization_agent=self.data_visualization_agent,
        )

    def process_query(self, user_question):
        """Process a user's natural language query and return the analysis result.

        Args:
            user_question (str): The user's query about the dataset.

        Returns:
            dict: A dictionary with 'type' (chart, table, or error) and 'data' or 'message'.
                  - For charts: {"type": "chart", "data": plot_json_string}
                  - For tables: {"type": "table", "data": list_of_dicts}
                  - For errors: {"type": "error", "message": error_message}

        """
        try:
            # Invoke the agent with the user's question and dataset
            response = self.pandas_data_analyst.invoke_agent(
                user_instructions=user_question,
                data_raw=self.df,
            )

            if not response:
                return {"type": "error", "message": "No response from the agent"}

            result = self.pandas_data_analyst.get_response()

            if not result:
                return {"type": "error", "message": "No result from the agent"}

            routing = result.get("routing_preprocessor_decision") if result else None

            # Handle chart output
            if routing == "chart" and not result.get("plotly_error", False):
                plot_data = result.get("plotly_graph")
                if plot_data:
                    return {"type": "chart", "data": plot_data}
                return {"type": "error", "message": "No valid chart data returned"}

            # Handle table output or fallback from chart errors
            data_wrangled = result.get("data_wrangled") if result else None
            if data_wrangled is not None:
                if not isinstance(data_wrangled, pd.DataFrame):
                    data_wrangled = pd.DataFrame(data_wrangled)
                data_list = data_wrangled.to_dict(orient="records")
                return {"type": "table", "data": data_list}

            # If neither chart nor table is available
            return {"type": "error", "message": "No data returned by the agent"}

        except Exception as e:
            return {"type": "error", "message": f"Error processing query: {str(e)}"}
