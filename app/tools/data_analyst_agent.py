import logging
import os

from typing import Any

import dotenv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from langchain.chat_models.base import BaseChatModel
from langchain.schema import HumanMessage, SystemMessage


# Add statsmodels for advanced statistical analysis
try:
    import statsmodels.api as sm

    HAVE_STATSMODELS = True
except ImportError:
    HAVE_STATSMODELS = False

dotenv.load_dotenv()


class DataVisualizationAgent:
    """An agent that uses AI to create data visualizations from natural language instructions.

    This agent takes a dataset and natural language instructions to generate
    an appropriate Plotly visualization along with the code used to create it.
    """

    def __init__(
        self,
        model: BaseChatModel,
        log: bool = False,
        log_path: str | None = None,
        system_prompt_template: str | None = None,
    ):
        """Initialize the DataVisualizationAgent.

        Args:
            model: The LLM to use for generating visualizations
            log: Whether to log the agent's operations
            log_path: Path to store logs (if logging is enabled)
            system_prompt_template: Custom system prompt to override the default

        """
        self.model = model
        self.log = log
        self.log_path = log_path if log_path else os.path.join(os.getcwd(), "logs/")
        self._setup_logging()

        # Default system prompt if none provided
        self.system_prompt_template = system_prompt_template or self._get_default_system_prompt()

        # Store the response and visualization data
        self.response = {}
        self.visualization_code = None
        self.plotly_figure = None

    def _setup_logging(self) -> None:
        """Configure logging if enabled."""
        if self.log:
            os.makedirs(self.log_path, exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(self.log_path, "data_visualization_agent.log"),
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            self.logger = logging.getLogger("DataVisualizationAgent")
            self.logger.info("Initialized DataVisualizationAgent")

    def _get_default_system_prompt(self) -> str:
        """Return the default system prompt for the visualization agent."""
        return """
        You are an expert data visualization AI.

        Your job is to create beautiful and insightful Plotly visualizations based on pandas DataFrames and user instructions.
        # Guidelines:
        1. Use Plotly Express (px) for simple charts and Plotly Graph Objects (go) for more complex or custom visualizations
        2. Use go.Figure() for creating figures, not px.Figure()
        3. Do not use any libraries that aren't already imported - only pandas, plotly.express, and plotly.graph_objects
        4. Create visualizations that provide clear insights into the data
        5. Include appropriate titles, labels, and color schemes
        6. Add interactivity when appropriate (hover information, dropdown filters)
        7. Handle different data types appropriately
        8. Format your response as a JSON object with two keys: "code" and "explanation"

        # Response Format Example:
        ```json
        {
          "code": "# Python code to create the visualization\\nimport plotly.express as px\\nfig = px.bar(df, x='category', y='value')\\nfig.update_layout(title='My Visualization')\\nfig",
          "explanation": "This visualization shows the relationship between categories and their values using a bar chart."
        }
        ```

        The visualization code should:
        1. Be valid Python code that uses the pandas DataFrame (df) and Plotly
        2. Return a plotly.graph_objects.Figure object
        3. Be designed to address the user's specific visualization request
        """

    def _format_data_summary(self, df: pd.DataFrame) -> str:
        """Create a summary of the dataframe to give the LLM context.

        Args:
            df: The pandas DataFrame to summarize

        Returns:
            A string containing the data summary

        """
        summary = []

        # Basic dataframe info
        summary.append(f"DataFrame Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Column information
        summary.append("\nColumn Information:")
        for col in df.columns:
            dtype = df[col].dtype
            missing = df[col].isna().sum()
            missing_pct = missing / len(df) * 100

            unique_values = df[col].nunique()
            unique_pct = unique_values / len(df) * 100  # noqa: F841

            if df[col].dtype in ["int64", "float64"]:
                min_val = df[col].min()
                max_val = df[col].max()
                mean_val = df[col].mean()
                value_info = f"min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.2f}"
            else:
                if unique_values <= 10:  # For categorical with few unique values
                    value_counts = df[col].value_counts().head(5).to_dict()
                    value_info = f"top values={value_counts}"
                else:
                    value_info = f"unique values={unique_values}"

            summary.append(f"- {col}: {dtype}, missing={missing} ({missing_pct:.1f}%), {value_info}")

        return "\n".join(summary)

    def generate_visualization(self, data: pd.DataFrame, instructions: str, max_retries: int = 3) -> dict[str, Any]:
        """Generate a visualization based on user instructions.

        Args:
            data: The pandas DataFrame to visualize
            instructions: Natural language instructions for the visualization
            max_retries: Maximum number of attempts to generate working code

        Returns:
            A dictionary containing the response from the agent

        """
        # Reset stored results
        self.response = {}
        self.visualization_code = None
        self.plotly_figure = None

        # Create data summary for context
        data_summary = self._format_data_summary(data)

        # Log the request
        if self.log:
            self.logger.info(f"Visualization request: {instructions}")
            self.logger.info(f"Data summary: {data_summary[:500]}...")

        # Create messages for the LLM
        messages = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(
                content=f"""
            # Data Summary
            {data_summary}

            # User Instructions
            {instructions}

            Generate a Plotly visualization that addresses these instructions.
            """
            ),
        ]

        # Get the initial response
        llm_response = self.model.invoke(messages)

        # Try to extract JSON content
        try:
            import json
            import re

            # Look for JSON pattern in the response
            json_match = re.search(r"```json\n(.*?)\n```", llm_response.content, re.DOTALL)
            response_json = json.loads(json_match.group(1)) if json_match else json.loads(llm_response.content)

            self.response = response_json
            self.visualization_code = response_json.get("code", "")

        except json.JSONDecodeError:
            # If JSON parsing fails, extract the code using regex
            self.response = {"explanation": llm_response.content}
            code_match = re.search(r"```python\n(.*?)\n```", llm_response.content, re.DOTALL)
            if code_match:
                self.visualization_code = code_match.group(1)
            else:
                self.visualization_code = ""

        # Execute the visualization code
        return self._execute_visualization_code(data, max_retries)

    def _execute_visualization_code(  # noqa: C901
        self, data: pd.DataFrame, max_retries: int
    ) -> dict[str, Any]:
        """Execute the generated visualization code with retry logic for errors.

        Args:
            data: The pandas DataFrame to visualize
            max_retries: Maximum number of attempts to generate working code

        Returns:
            The response dictionary with the final result

        """
        retry_count = 0
        success = False
        error_message = ""

        while not success and retry_count < max_retries:
            try:
                # Create a safe execution environment
                local_vars = {
                    "df": data,
                    "px": px,
                    "go": go,
                    "pd": pd,
                    "Figure": go.Figure,  # Add Figure directly to avoid confusion
                }

                # Add statsmodels if available
                if HAVE_STATSMODELS:
                    local_vars["sm"] = sm

                # Execute the code
                exec(self.visualization_code, globals(), local_vars)

                # Look for a figure object in the local variables
                for var_name, var_value in local_vars.items():  # noqa: B007
                    if isinstance(var_value, go.Figure):
                        self.plotly_figure = var_value
                        success = True
                        break

                if not success:
                    error_message = "No Plotly figure found in the generated code output."
                    retry_count += 1
                    self._request_code_fix(data, error_message)

            except Exception as e:
                error_message = str(e)
                if self.log:
                    self.logger.error(
                        f"Error executing visualization code (attempt {retry_count + 1}): {error_message}"
                    )

                retry_count += 1
                if retry_count < max_retries:
                    # Add explicit help for known errors in the error message to guide the LLM
                    if "No module named 'statsmodels'" in error_message:
                        error_message += ". Please avoid using statsmodels and use only pandas, plotly.express, and plotly.graph_objects."
                    elif "has no attribute 'Figure'" in error_message:
                        error_message += ". Use go.Figure() for creating figures, not px.Figure()."

                    self._request_code_fix(data, error_message)

        # Update the response with execution results
        self.response["success"] = success
        if not success:
            self.response["error"] = error_message

        return self.response

    def _request_code_fix(self, data: pd.DataFrame, error_message: str) -> None:
        """Request a fix for the visualization code that produced an error.

        Args:
            data: The pandas DataFrame to visualize
            error_message: The error message from the failed execution

        """
        data_summary = self._format_data_summary(data)

        messages = [
            SystemMessage(content=self.system_prompt_template),
            HumanMessage(
                content=f"""
            The previous visualization code produced an error:

            ERROR: {error_message}

            Here is the code that needs to be fixed:
            ```python
            {self.visualization_code}
            ```

            # Data Summary
            {data_summary}

            Please fix the code and provide a working Plotly visualization. Use only pandas, plotly.express (px), and plotly.graph_objects (go).
            Remember to:
            1. Use go.Figure() for creating figures, not px.Figure()
            2. Do not use statsmodels or any other libraries not already imported
            3. For pie charts use px.pie(), for scatter plots use px.scatter(), etc.
            4. For complex figures, create a go.Figure() and add traces with fig.add_trace()
            """
            ),
        ]

        # Get the response for fixing the code
        llm_response = self.model.invoke(messages)

        # Try to extract JSON content
        try:
            import json
            import re

            # Look for JSON pattern in the response
            json_match = re.search(r"```json\n(.*?)\n```", llm_response.content, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group(1))
                self.response = response_json
                self.visualization_code = response_json.get("code", "")
            else:
                # Extract the code using regex
                code_match = re.search(r"```python\n(.*?)\n```", llm_response.content, re.DOTALL)
                if code_match:
                    self.visualization_code = code_match.group(1)
                    self.response["code"] = self.visualization_code

        except (json.JSONDecodeError, AttributeError):
            # If JSON parsing fails, extract the code using regex
            code_match = re.search(r"```python\n(.*?)\n```", llm_response.content, re.DOTALL)
            if code_match:
                self.visualization_code = code_match.group(1)
                self.response["code"] = self.visualization_code

    def get_plotly_figure(self) -> go.Figure | None:
        """Get the generated Plotly figure.

        Returns:
            The Plotly figure object, or None if no figure was generated

        """
        return self.plotly_figure

    def get_response(self) -> dict[str, Any]:
        """Get the full response from the agent.

        Returns:
            The response dictionary

        """
        return self.response

    def get_visualization_code(self, format_markdown: bool = False) -> str:
        """Get the code used to generate the visualization.

        Args:
            format_markdown: Whether to format the code as markdown

        Returns:
            The visualization code as a string

        """
        if not self.visualization_code:
            return ""

        if format_markdown:
            return f"```python\n{self.visualization_code}\n```"

        return self.visualization_code


# Example usage
if __name__ == "__main__":
    import os

    from langchain_openai import ChatOpenAI

    # Set up the model
    MODEL = "gpt-4o-mini"
    LOG_PATH = os.path.join(os.getcwd(), "logs/")

    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

    llm = ChatOpenAI(model=MODEL)

    # Load example dataset
    df = pd.read_csv(
        "https://raw.githubusercontent.com/business-science/ai-data-science-team/refs/heads/master/data/churn_data.csv"
    )

    # Create agent and generate visualization
    vis_agent = DataVisualizationAgent(model=llm, log=True, log_path=LOG_PATH)

    response = vis_agent.generate_visualization(
        data=df,
        instructions="Make a boxplot, pie chart, histogram, also create a table of all metrics, of the monthly charges vs Churn. Churn should be on X axis and distribution on Y axis.",
    )

    # Get and display the results
    fig = vis_agent.get_plotly_figure()
    if fig:
        fig.show()
