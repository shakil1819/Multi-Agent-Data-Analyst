import glob
import os

import pandas as pd
import sqlalchemy as sql

from ai_data_science_team import (
    SQLDatabaseAgent,
)
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


load_dotenv()


class SqlDataAnalystAgent:
    def __init__(self, model_name="gpt-4o-mini"):
        """Initialize the DataAnalystAgent with an OpenAI API key and model.

        This agent loads a CSV dataset from the ./data directory into an in-memory
        SQLite database and sets up the SQLDatabaseAgent for querying.

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

        # Load the CSV file from the ./data directory
        csv_files = glob.glob("./data/*.csv")
        if len(csv_files) != 1:
            raise ValueError("Expected exactly one CSV file in ./data directory")
        self.df = pd.read_csv(csv_files[0])

        # Create an in-memory SQLite database and load the dataframe into it
        self.engine = sql.create_engine("sqlite:///:memory:")
        self.df.to_sql("data", self.engine, index=False)

        # Set up the SQLDatabaseAgent with the in-memory database connection
        self.sql_db_agent = SQLDatabaseAgent(
            model=self.llm,
            connection=self.engine.connect(),
            n_samples=1,
            log=False,
            bypass_recommended_steps=True,
        )

    async def process_query(self, user_question):
        """Process a user's natural language query and return the SQL query and result.

        This method uses the SQLDatabaseAgent to interpret the user's question,
        generate an SQL query, execute it on the in-memory database, and return
        the result.

        Args:
            user_question (str): The user's query about the dataset.

        Returns:
            dict: A dictionary containing the processing status and results.
                - On success: {"status": "success", "sql_query": str, "data": list_of_dicts}
                - On error: {"status": "error", "message": str}

        """
        try:
            # Invoke the agent to process the user's question
            await self.sql_db_agent.ainvoke_agent(user_instructions=user_question)

            # Retrieve the generated SQL query and the resulting dataframe
            sql_query = self.sql_db_agent.get_sql_query_code()
            response_df = self.sql_db_agent.get_data_sql()

            if response_df is not None:
                # Convert the dataframe to a list of dictionaries for easy serialization
                data = response_df.to_dict(orient="records")
                return {"status": "success", "sql_query": sql_query, "data": data}
            return {"status": "error", "message": "No data returned from the query"}

        except Exception as e:
            # Capture and return any errors that occur during processing
            return {"status": "error", "message": f"Error processing query: {str(e)}"}
