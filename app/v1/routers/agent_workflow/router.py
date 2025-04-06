from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.v1.routers.agent_workflow.graph import AgentWorkflowGraph


router = APIRouter()
agent_workflow = AgentWorkflowGraph()


class QueryRequest(BaseModel):
    """Request model for agent workflow queries."""

    query: str


class QueryResponse(BaseModel):
    """Response model for agent workflow results."""

    result: dict
    agent_used: str


@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    """Process a user query through the multi-agent workflow.

    The workflow will automatically select the appropriate agent based on the query.
    """
    try:
        return await agent_workflow.process_query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}") from e
