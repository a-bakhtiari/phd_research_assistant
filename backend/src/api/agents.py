"""Agents API endpoints - handles AI agent tasks."""

from fastapi import APIRouter

router = APIRouter()


@router.post("/discover")
async def start_paper_discovery():
    """Start an agent-based paper discovery task."""
    return {"task_id": "sample-task-id"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of an agent task."""
    return {"status": "running", "progress": 0.5}
