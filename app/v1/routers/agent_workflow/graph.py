from typing import Any, Literal, Optional  # noqa: UP035

from pydantic import BaseModel
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from app.tools.data_analyst_agent import DataAnalystAgent
from app.tools.sql_data_analyst_agent import SqlDataAnalystAgent
from app.v1.routers.agent_workflow.persistence import ChromaDBStatePersistence, get_chroma_client


class AgentState(BaseModel):
    """State model for the agent workflow graph."""

    query: str
    result: Optional[dict[str, Any]] = None
    agent_used: Optional[str] = None


class QueryClassifier(BaseNode[AgentState]):
    """Node to classify the query and determine which agent to use."""

    async def run(self, ctx: GraphRunContext[AgentState]) -> Literal["use_sql_agent", "use_data_agent"]:
        """Classify the query to determine which agent to use.

        Returns:
            str: Either "use_sql_agent" or "use_data_agent"

        """
        # Simple classifier based on keywords
        query = ctx.state.query.lower()

        # Keywords that suggest SQL operations
        sql_keywords = [
            "sql",
            "query",
            "table",
            "database",
            "select",
            "join",
            "where",
            "group by",
            "order by",
            "having",
            "count",
            "sum",
            "average",
            "filter",
            "find records",
            "search records",
        ]

        # Keywords that suggest data analysis or visualization
        data_keywords = [
            "analyze",
            "chart",
            "graph",
            "plot",
            "visualization",
            "trend",
            "pattern",
            "correlation",
            "distribution",
            "histogram",
            "scatter plot",
            "bar chart",
            "pie chart",
            "dashboard",
        ]

        # Count matches for each category
        sql_score = sum(1 for keyword in sql_keywords if keyword in query)
        data_score = sum(1 for keyword in data_keywords if keyword in query)

        # Determine which agent to use based on keyword matches
        if sql_score > data_score:
            return "use_sql_agent"
        return "use_data_agent"


class UseSqlAgent(BaseNode[AgentState]):
    """Node to process the query using the SQL Data Analyst Agent."""

    async def run(self, ctx: GraphRunContext[AgentState]) -> End[dict[str, Any]]:
        """Process the query using the SQL Data Analyst Agent.

        Returns:
            End: The result from the SQL Data Analyst Agent

        """
        sql_agent = SqlDataAnalystAgent()
        result = await sql_agent.process_query(ctx.state.query)

        # Update the state
        ctx.state.result = result
        ctx.state.agent_used = "sql_data_analyst"

        # Return the result and end the graph execution
        return End({"result": result, "agent_used": "sql_data_analyst"})


class UseDataAgent(BaseNode[AgentState]):
    """Node to process the query using the Data Analyst Agent."""

    async def run(self, ctx: GraphRunContext[AgentState]) -> End[dict[str, Any]]:
        """Process the query using the Data Analyst Agent.

        Returns:
            End: The result from the Data Analyst Agent

        """
        data_agent = DataAnalystAgent()
        result = data_agent.process_query(ctx.state.query)

        # Update the state
        ctx.state.result = result
        ctx.state.agent_used = "data_analyst"

        # Return the result and end the graph execution
        return End({"result": result, "agent_used": "data_analyst"})


class AgentWorkflowGraph:
    """Multi-agent workflow graph implementation using pydantic-ai and ChromaDB.

    This class manages the workflow that selects and runs the appropriate agent
    based on the user's query.
    """

    def __init__(self):
        """Initialize the agent workflow graph."""
        # Create the graph with nodes and edges
        nodes = {
            "start": QueryClassifier(),
            "use_sql_agent": UseSqlAgent(),
            "use_data_agent": UseDataAgent(),
        }

        edges = {
            "start": {
                "use_sql_agent": "use_sql_agent",
                "use_data_agent": "use_data_agent",
            }
        }

        # Create the graph using named parameters
        self.graph = Graph(
            state_type=AgentState,
            nodes=nodes,
            edges=edges,
        )

        # Set up persistence with ChromaDB
        chroma_client = get_chroma_client()
        self.persistence = ChromaDBStatePersistence(
            chroma_client=chroma_client,
            collection_name="agent_workflow_states",
        )

    async def process_query(self, query: str) -> dict[str, Any]:
        """Process a user query through the multi-agent workflow.

        Args:
            query: The user's query string

        Returns:
            dict: A dictionary containing the result and the agent used

        """
        # Initialize the state with the query
        state = AgentState(query=query)

        # Run the graph with the state
        return await self.graph.arun(state, state_persistence=self.persistence)
