import os
import re

from typing import Any

import dotenv
import pandas as pd
import plotly.graph_objects as go
import sqlalchemy as sql

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


dotenv.load_dotenv()


class SQLDataAnalysisAgent:
    """A single agent that handles SQL database querying and data visualization."""

    def __init__(
        self,
        model: BaseChatModel,
        connection: sql.engine.base.Connection = None,
        engine_url: str = None,
        n_samples: int = 5,
        log: bool = False,
        log_path: str = None,
        verbose: bool = False,
    ):
        """Initialize the SQL Data Analysis Agent."""
        self.model = model
        self.n_samples = n_samples
        self.log = log
        self.log_path = log_path
        self.verbose = verbose

        # State management
        self._state = {
            "sql_query_code": None,
            "sql_database_function": None,
            "data_sql": None,
            "data_visualization_function": None,
            "plotly_graph": None,
            "error": None,
        }

        # Set up database connection
        if connection is None and engine_url is None:
            raise ValueError("Either connection or engine_url must be provided")

        self.connection = connection
        if connection is None:
            engine = sql.create_engine(engine_url)
            self.connection = engine.connect()

        # Create log directory if needed
        if self.log and self.log_path:
            os.makedirs(self.log_path, exist_ok=True)

    def invoke_agent(self, user_instructions: str, auto_display: bool = True) -> dict[str, Any]:
        """Process user instructions and generate responses."""
        # Reset state for new query
        self._reset_state()

        # Determine analysis type and execute steps
        analysis_type = self._determine_analysis_type(user_instructions)
        self._generate_and_execute_sql(user_instructions, analysis_type)

        # Create visualization if needed
        if (
            analysis_type["needs_visualization"]
            and self._state["data_sql"] is not None
            and not isinstance(self._state["data_sql"], str)
        ):
            self._generate_visualization(user_instructions)

        # Auto-display results if requested
        if auto_display:
            self.display_results()

        return self._state

    def display_results(self):
        """Display only metrics and visualizations."""
        # Check for errors first
        if self._state.get("error"):
            print(f"\n--- Error ---\n{self._state['error']}")
            return

        # Display data metrics
        df = self.get_data_sql()
        if df is not None:
            if isinstance(df, str):
                print(f"\n--- Error ---\n{df}")
                return

            print("\n--- Data Metrics ---")

            # Show basic DataFrame information
            print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

            # For numerical columns, show summary statistics
            num_cols = df.select_dtypes(include=["number"]).columns
            if len(num_cols) > 0:
                print("\nNumerical Summary:")
                print(df[num_cols].describe().round(2))

            # Show sample data (just a few rows)
            print("\nSample Data:")
            pd.set_option("display.max_columns", 10)
            pd.set_option("display.width", 1000)
            print(df.head(min(5, len(df))))

        # Display visualization if available
        fig = self.get_plotly_graph()
        if fig:
            try:
                # For Jupyter environments
                from IPython.display import display

                display(fig)
            except ImportError:
                # For script environments, save and open in browser
                if not os.path.exists("visualizations"):
                    os.makedirs("visualizations")
                html_path = "visualizations/analysis_result.html"
                fig.write_html(html_path)
                print(f"\nVisualization saved to: {html_path}")
                try:
                    import webbrowser

                    webbrowser.open("file://" + os.path.realpath(html_path))
                except Exception as e:
                    if self.verbose:
                        print(f"Could not open browser: {e}")

    def _determine_analysis_type(self, user_instructions: str) -> dict[str, bool]:
        """Determine what kind of analysis is needed."""
        visualization_keywords = [
            "plot",
            "chart",
            "graph",
            "visualize",
            "visualization",
            "show",
            "display",
            "dashboard",
        ]

        needs_visualization = any(keyword in user_instructions.lower() for keyword in visualization_keywords)

        return {"needs_visualization": needs_visualization}

    def _extract_code_from_response(self, response: str) -> str:
        """Extract clean code from LLM responses that might contain markdown or explanations."""
        # Look for Python code blocks
        code_block_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", response)
        if code_block_match:
            return code_block_match.group(1).strip()

        # If no code block found, return the entire response
        return response.strip()

    def _generate_and_execute_sql(self, user_instructions: str, analysis_type: dict[str, bool]) -> None:  # noqa: ARG002
        """Generate SQL query based on user instructions and execute it."""
        tables_info = self._get_database_schema()

        # Generate SQL query using LLM
        sql_prompt = f"""
        You are an expert SQL developer. Given the following database schema:

        {tables_info}

        Generate a SQL query to answer this question:
        {user_instructions}

        Return ONLY the SQL query without any explanations or markdown formatting.
        Do not include backticks (```) or 'sql' at the beginning or end.
        """

        sql_result = self.model.invoke(sql_prompt)
        sql_query = self._extract_code_from_response(sql_result.content)
        self._state["sql_query_code"] = sql_query

        # Log the SQL query
        if self.verbose:
            print(f"Generated SQL Query: {sql_query}")

        # Generate Python function that executes this SQL
        function_prompt = f"""
        Create a Python function that executes the following SQL query using SQLAlchemy and returns a pandas DataFrame:

        ```sql
        {sql_query}
        ```

        The function should:
        1. Be named 'execute_sql_query'
        2. Accept a SQLAlchemy connection object as its only parameter
        3. Execute the SQL query directly using connection.execute()
        4. Return the results as a pandas DataFrame using pd.DataFrame(result.fetchall(), columns=result.keys())

        Return ONLY the Python code without any explanations or markdown. Do not include backticks.
        """

        function_result = self.model.invoke(function_prompt)
        sql_function = self._extract_code_from_response(function_result.content)
        self._state["sql_database_function"] = sql_function

        if self.verbose:
            print(f"Generated Function: {sql_function}")

        # Execute the SQL query directly for more control and better error handling
        try:
            # Direct execution approach
            result = self.connection.execute(sql.text(sql_query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            self._state["data_sql"] = df
        except Exception as direct_error:
            try:
                # Fallback to LLM-generated function if direct execution fails
                local_namespace = {"pd": pd, "sql": sql}

                # Validate the function before execution
                if "execute_sql_query" not in sql_function:
                    raise ValueError("Function name 'execute_sql_query' not found in generated code")

                # Execute the function code to define it
                exec(sql_function, globals(), local_namespace)

                # Get the function from the namespace
                execute_sql_query = local_namespace.get("execute_sql_query")
                if not execute_sql_query:
                    raise ValueError("Could not find execute_sql_query function after execution")

                # Call the function with the connection
                df = execute_sql_query(self.connection)
                self._state["data_sql"] = df
            except Exception as e:
                error_msg = f"SQL execution failed: {str(e)}\nOriginal error: {str(direct_error)}"
                self._state["error"] = error_msg
                self._state["data_sql"] = pd.DataFrame({"Error": [error_msg]})

    def _generate_visualization(self, user_instructions: str) -> None:
        """Generate visualization based on query results and user instructions."""
        if self._state["data_sql"] is None or isinstance(self._state["data_sql"], str):
            return

        df = self._state["data_sql"]

        # Generate visualization code using LLM
        vis_prompt = f"""
        You are an expert data visualization developer using Plotly.

        Given the following DataFrame (showing first {min(self.n_samples, len(df))} rows):

        {df.head(self.n_samples).to_string()}

        DataFrame columns and types:
        {df.dtypes.to_string()}

        Create a Python function that generates an appropriate Plotly visualization based on this user request:
        {user_instructions}

        The function should:
        1. Be named 'create_visualization'
        2. Accept a pandas DataFrame as input named 'df'
        3. Return a plotly.graph_objects.Figure object
        4. Include interactive elements when appropriate (dropdowns, hover info, etc.)
        5. Use appropriate colors and styling
        6. Set appropriate layout margins, titles and sizing

        Return ONLY the Python function code without any explanations or markdown formatting.
        Do not include backticks (```) or 'python' at the beginning or end.
        """

        vis_result = self.model.invoke(vis_prompt)
        vis_function = self._extract_code_from_response(vis_result.content)
        self._state["data_visualization_function"] = vis_function

        if self.verbose:
            print(f"Generated Visualization Function: {vis_function}")

        # Execute the visualization function
        try:
            # Set up environment
            local_namespace = {"pd": pd, "go": go, "np": None, "plt": None}

            # Import optional dependencies
            try:
                import plotly.express as px

                local_namespace["px"] = px
            except ImportError:
                pass

            try:
                import numpy as np

                local_namespace["np"] = np
            except ImportError:
                pass

            try:
                import matplotlib.pyplot as plt

                local_namespace["plt"] = plt
            except ImportError:
                pass

            # Validate function before execution
            if "create_visualization" not in vis_function:
                raise ValueError("Function name 'create_visualization' not found in generated code")

            # Execute the function code to define it
            exec(vis_function, globals(), local_namespace)

            # Get the function from the namespace
            create_visualization = local_namespace.get("create_visualization")
            if not create_visualization:
                raise ValueError("Could not find create_visualization function after execution")

            # Call the function with the DataFrame
            fig = create_visualization(df)
            self._state["plotly_graph"] = fig
        except Exception as e:
            error_msg = f"Visualization creation failed: {str(e)}"
            self._state["error"] = error_msg
            if self.verbose:
                print(error_msg)

    def _get_database_schema(self) -> str:
        """Get database schema information."""
        inspector = sql.inspect(self.connection.engine)
        tables = inspector.get_table_names()

        schema_info = []
        for table in tables:
            columns = inspector.get_columns(table)
            col_info = [f"    - {col['name']}: {col['type']}" for col in columns]
            schema_info.append(f"Table: {table}\nColumns:\n" + "\n".join(col_info))

        return "\n\n".join(schema_info)

    def _reset_state(self) -> None:
        """Reset the agent state for a new query."""
        self._state = {
            "sql_query_code": None,
            "sql_database_function": None,
            "data_sql": None,
            "data_visualization_function": None,
            "plotly_graph": None,
            "error": None,
        }

    def _log_interaction(self, step: str, content: str) -> None:
        """Log interaction to file if logging is enabled."""
        if not self.log or not self.log_path:
            return

        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{step}.txt"
        filepath = os.path.join(self.log_path, filename)

        with open(filepath, "w") as f:
            f.write(content)

    # Public methods to access state
    def get_state_keys(self) -> list:
        """Return the keys available in the agent state."""
        return list(self._state.keys())

    def get_sql_query_code(self, markdown: bool = False) -> str:
        """Get the SQL query code."""
        sql_code = self._state.get("sql_query_code")
        if sql_code and markdown:
            return f"```sql\n{sql_code}\n```"
        return sql_code

    def get_sql_database_function(self, markdown: bool = False) -> str:
        """Get the SQL database function."""
        function_code = self._state.get("sql_database_function")
        if function_code and markdown:
            return f"```python\n{function_code}\n```"
        return function_code

    def get_data_sql(self) -> pd.DataFrame | None:
        """Get the DataFrame from SQL query execution."""
        return self._state.get("data_sql")

    def get_data_visualization_function(self, markdown: bool = False) -> str:
        """Get the data visualization function."""
        function_code = self._state.get("data_visualization_function")
        if function_code and markdown:
            return f"```python\n{function_code}\n```"
        return function_code

    def get_plotly_graph(self) -> go.Figure | None:
        """Get the Plotly graph object."""
        return self._state.get("plotly_graph")

    def get_error(self) -> str | None:
        """Get any error that occurred during processing."""
        return self._state.get("error")


# Example usage
if __name__ == "__main__":
    # Setup
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")  # Replace with your API key

    # Initialize LLM and database connection
    llm = ChatOpenAI(model="gpt-4o")
    engine = sql.create_engine("sqlite:///northwind.db")
    conn = engine.connect()

    # Create the agent with verbose mode for debugging
    agent = SQLDataAnalysisAgent(
        model=llm,
        connection=conn,
        n_samples=5,
        log=False,
        verbose=True,  # Set to True for debugging
    )

    # Example 1: Get database tables
    print("\n==== QUERY 1: Database Tables ====")
    agent.invoke_agent("What tables are in the database?")

    # Example 2: Visualization
    print("\n==== QUERY 2: Sales Visualization ====")
    agent.invoke_agent(
        "Make a plot of sales revenue by month by territory. Make a dropdown for the user to select the territory."
    )

    # Close connection when done
    conn.close()
