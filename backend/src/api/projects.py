"""
Projects API endpoints.

Handles project management operations including CRUD operations for research projects.
"""

from typing import List, Annotated, Optional
from fastapi import APIRouter, HTTPException, Depends, status

from src.models.schemas import ProjectCreate, ProjectResponse, ProjectStats
from src.services import ProjectService
from src.dependencies import get_project_service

router = APIRouter()


@router.get("/", response_model=List[ProjectResponse])
async def list_projects(
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    List all research projects.

    Returns:
        List[ProjectResponse]: List of all projects
    """
    return project_service.get_all_projects()


@router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project: ProjectCreate,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    Create a new research project.

    Args:
        project: Project creation data

    Returns:
        ProjectResponse: Created project data
    """
    import logging
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"Creating project with name: {project.name}")
        result = project_service.create_project(
            name=project.name,
            description=project.description
        )
        logger.info(f"Successfully created project: {result.name}")
        return result
    except ValueError as e:
        logger.error(f"ValueError creating project: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error creating project: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    Get a specific project by ID.

    Args:
        project_id: Project ID

    Returns:
        ProjectResponse: Project data
    """
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return project


@router.get("/{project_id}/stats", response_model=ProjectStats)
async def get_project_stats(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    Get statistics for a project.

    Args:
        project_id: Project ID

    Returns:
        ProjectStats: Project statistics
    """
    stats = project_service.get_project_stats(project_id)
    if not stats:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return stats


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    Delete a project.

    Args:
        project_id: Project ID
    """
    success = project_service.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return None
