import os

from dataclasses import dataclass
from typing import Any, TypedDict

import pandas as pd
import sqlalchemy as sql

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import UsageLimits

from app.tools.data_analyst_agent import DataVisualizationAgent
from app.tools.sql_data_analyst_agent import SQLDataAnalysisAgent


# Define our dependency type for orchestration
@dataclass
class OrchestratorDependency:
    """Dependencies for the orchestrator agent."""

    user_prompt: str
    model: BaseChatModel
    data: pd.DataFrame | None = None
    db_connection: sql.engine.base.Connection | None = None
    usage_limits: UsageLimits | None = None


# Type for streamed results
class AnalysisResult(TypedDict):
    success: bool
    message: str
    visualization_path: str | None
    error: str | None
    data_summary: dict[str, Any] | None


# Create our master orchestrator agent
orchestrator_agent = Agent(
    "openai:gpt-4o",
    deps_type=OrchestratorDependency,
    result_type=AnalysisResult,
    system_prompt="""
    You are an expert data analysis orchestrator. Your job is to:
    1. Understand user requests related to data analysis and visualization
    2. Determine whether to use SQL database analysis or direct DataFrame analysis
    3. Call the appropriate agent to handle the request
    4. Return results in a clear, organized manner

    For SQL database requests, use the sql_agent tool.
    For DataFrame visualization requests, use the visualization_agent tool.
""",
)


@orchestrator_agent.tool
async def sql_agent(ctx: RunContext[OrchestratorDependency], query: str) -> dict[str, Any]:  # noqa: D417
    """Process a SQL database query and visualization request.

    Args:
        query: The user's analysis request/question about the database

    Returns:
        A dictionary with the analysis results

    """
    if ctx.deps.db_connection is None:
        return {"error": "Database connection is required but not provided"}

    # Initialize the SQL agent with the provided connection
    sql_agent = SQLDataAnalysisAgent(
        model=ctx.deps.model, connection=ctx.deps.db_connection, n_samples=5, log=True, log_path="logs/", verbose=True
    )

    # Execute the query but don't auto-display (we'll handle that)
    results = sql_agent.invoke_agent(query, auto_display=False)

    # Check for errors
    if results.get("error"):
        return {"success": False, "error": results.get("error"), "message": f"Analysis failed: {results.get('error')}"}

    # Get visualization path if available
    vis_path = None
    if results.get("plotly_graph"):
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        vis_path = "visualizations/analysis_result.html"
        results.get("plotly_graph").write_html(vis_path)

    # Get data summary
    data_summary = None
    df = sql_agent.get_data_sql()
    if df is not None and not isinstance(df, str):
        data_summary = {
            "shape": df.shape,
            "columns": list(df.columns),
            "sample": df.head(5).to_dict() if len(df) > 0 else {},
        }

    return {
        "success": True,
        "message": "SQL analysis completed successfully",
        "visualization_path": vis_path,
        "sql_query": sql_agent.get_sql_query_code(),
        "data_summary": data_summary,
    }


@orchestrator_agent.tool
async def visualization_agent(ctx: RunContext[OrchestratorDependency], instructions: str) -> dict[str, Any]:  # noqa: D417
    """Create a visualization from a DataFrame based on instructions.

    Args:
        instructions: The visualization instructions

    Returns:
        A dictionary with the visualization results

    """
    if ctx.deps.data is None:
        return {"error": "DataFrame is required but not provided"}

    # Initialize the visualization agent
    vis_agent = DataVisualizationAgent(model=ctx.deps.model, log=True, log_path="logs/")

    # Generate the visualization
    response = vis_agent.generate_visualization(data=ctx.deps.data, instructions=instructions)

    # Check for errors
    if not response.get("success", False):
        return {
            "success": False,
            "error": response.get("error", "Unknown error"),
            "message": f"Visualization failed: {response.get('error', 'Unknown error')}",
        }

    # Save the visualization if available
    vis_path = None
    fig = vis_agent.get_plotly_figure()
    if fig:
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")
        vis_path = "visualizations/analysis_result.html"
        fig.write_html(vis_path)

    return {
        "success": True,
        "message": "Visualization created successfully",
        "visualization_path": vis_path,
        "visualization_code": vis_agent.get_visualization_code(),
        "explanation": response.get("explanation", ""),
    }


@orchestrator_agent.tool
async def determine_data_source(ctx: RunContext[OrchestratorDependency], query: str) -> str:  # noqa: D417
    """Determine whether to use SQL database or DataFrame analysis based on the query.

    Args:
        query: The user's analysis request/question

    Returns:
        A recommendation for which data source to use ("sql" or "dataframe")

    """
    # Check if we have both options available
    has_db = ctx.deps.db_connection is not None
    has_df = ctx.deps.data is not None

    # If we only have one option, use that
    if has_db and not has_df:
        return "sql"
    if has_df and not has_db:
        return "dataframe"

    # If we have both options, determine based on query content
    sql_keywords = ["sql", "database", "table", "query", "join", "select", "from", "where"]
    has_sql_keywords = any(keyword in query.lower() for keyword in sql_keywords)

    if has_sql_keywords:
        return "sql"
    return "dataframe"


async def process_user_input(
    user_input: str,
    data: pd.DataFrame = None,
    db_connection: sql.engine.base.Connection = None,
    usage_limits: UsageLimits = None,
) -> dict[str, Any]:
    """Process a user input with the orchestrator agent.

    Args:
        user_input: The user's prompt/question
        data: Optional DataFrame to analyze
        db_connection: Optional database connection
        usage_limits: Optional usage limits

    Returns:
        The results of the analysis

    """
    # Set up the LLM
    model = ChatOpenAI(model_name="gpt-4o")

    # Create dependencies
    deps = OrchestratorDependency(
        user_prompt=user_input, model=model, data=data, db_connection=db_connection, usage_limits=usage_limits
    )

    # Run the agent
    result = await orchestrator_agent.run(user_input, deps=deps, usage_limits=usage_limits)

    return result.data


async def run_agent_orchestrator(
    user_input: str, data_path: str = None, db_url: str = None, usage_limits: UsageLimits = None
) -> dict[str, Any]:
    """Run the agent orchestrator with file path or database URL.

    Args:
        user_input: The user's prompt/question
        data_path: Optional path to a data file (CSV, Excel)
        db_url: Optional database URL
        usage_limits: Optional usage limits

    Returns:
        The results of the analysis

    """
    data = None
    db_connection = None

    # Load data if provided
    if data_path:
        if data_path.endswith(".csv"):
            data = pd.read_csv(data_path)
        elif data_path.endswith((".xls", ".xlsx")):
            data = pd.read_excel(data_path)
        else:
            return {"error": "Unsupported file format. Please use .csv, .xls, or .xlsx"}

    # Set up database connection if provided
    if db_url:
        try:
            engine = sql.create_engine(db_url)
            db_connection = engine.connect()
        except Exception as e:
            return {"error": f"Failed to connect to database: {str(e)}"}

    try:
        # Process the request
        result = await process_user_input(
            user_input=user_input, data=data, db_connection=db_connection, usage_limits=usage_limits
        )

        # Clean up database connection if we created one
        if db_connection:
            db_connection.close()

        return result
    except Exception as e:
        if db_connection:
            db_connection.close()
        return {"error": str(e)}


# Streaming version of the process_user_input function
async def process_user_input_stream(
    user_input: str,
    data: pd.DataFrame = None,
    db_connection: sql.engine.base.Connection = None,
    usage_limits: UsageLimits = None,
):
    """Process a user input with the orchestrator agent and stream the results.

    Args:
        user_input: The user's prompt/question
        data: Optional DataFrame to analyze
        db_connection: Optional database connection
        usage_limits: Optional usage limits

    Returns:
        An async generator that yields progress updates

    """
    # Set up the LLM
    model = ChatOpenAI(model_name="gpt-4o")

    # Create dependencies
    deps = OrchestratorDependency(
        user_prompt=user_input, model=model, data=data, db_connection=db_connection, usage_limits=usage_limits
    )

    # First yield the starting message
    yield "Starting analysis...\n"

    try:
        # Run the agent and get the result (non-streaming first)
        run_result = await orchestrator_agent.run(user_input, deps=deps, usage_limits=usage_limits)

        # Yield progress updates
        yield "Processing data and creating visualization...\n"

        # Get the final result
        result = run_result.data

        # Yield the final result summary
        if result.get("success", False):
            yield "\nAnalysis completed successfully!\n"
            if result.get("visualization_path"):
                yield f"Visualization saved to: {result.get('visualization_path')}\n"
                yield "You can view the visualization in your browser.\n"

            if result.get("data_summary"):
                yield "\nData Summary:\n"
                shape = result.get("data_summary", {}).get("shape")
                if shape:
                    yield f"- Shape: {shape[0]} rows × {shape[1]} columns\n"

                columns = result.get("data_summary", {}).get("columns")
                if columns:
                    yield f"- Columns: {', '.join(columns)}\n"
        else:
            yield f"\nAnalysis failed: {result.get('error', 'Unknown error')}\n"

    except Exception as e:
        # Handle any exceptions
        yield f"\nError during analysis: {str(e)}\n"
