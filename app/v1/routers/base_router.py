from fastapi import APIRouter

from app.v1.routers.agent_workflow import router as agent_workflow_router
from app.v1.routers.users import users_router


router = APIRouter(prefix="/v1")
router.include_router(users_router.router, prefix="/users", tags=["Users"])
router.include_router(agent_workflow_router.router, prefix="/agent-workflow", tags=["Agent Workflow"])
