"""
FastAPI dependencies for dependency injection.

Provides reusable dependencies for services and managers.
"""

import logging
from pathlib import Path
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, Header

from src.config import get_settings
from src.core.database import DatabaseManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.services import (
    ProjectService,
    PaperService,
    ChatService,
    RecommendationService,
    SettingsService
)
from src.services.queue_service import QueueService

logger = logging.getLogger(__name__)

# Global instances (initialized once)
_project_service = None
_llm_manager = None
_prompt_manager = None
_vector_managers = {}  # Cache vector managers per project to avoid concurrent initialization


def get_project_service() -> ProjectService:
    """
    Get or create ProjectService instance.

    Returns:
        ProjectService instance
    """
    global _project_service
    if _project_service is None:
        # Get projects root - go up one level from backend/ to find projects/
        projects_root = Path.cwd().parent / "projects"
        if not projects_root.exists():
            # Fallback: if we're not in backend/, try current directory
            projects_root = Path.cwd() / "projects"
        _project_service = ProjectService(projects_root=projects_root)
    return _project_service


def get_llm_manager() -> LLMManager:
    """
    Get or create LLMManager instance.

    Returns:
        LLMManager instance
    """
    global _llm_manager
    if _llm_manager is None:
        settings = get_settings()
        # Build nested config structure that LLMManager expects
        config = {}

        if settings.openai_api_key:
            config["openai"] = {
                "api_key": settings.openai_api_key,
                "model": settings.openai_model or "gpt-3.5-turbo"
            }

        if settings.anthropic_api_key:
            config["anthropic"] = {
                "api_key": settings.anthropic_api_key,
                "model": settings.anthropic_model or "claude-3-sonnet-20240229"
            }

        if settings.deepseek_api_key:
            config["deepseek"] = {
                "api_key": settings.deepseek_api_key,
                "model": settings.deepseek_model or "deepseek-chat"
            }

        config["default_provider"] = settings.default_llm_provider

        _llm_manager = LLMManager(config=config)
    return _llm_manager


def get_prompt_manager() -> PromptManager:
    """
    Get or create PromptManager instance.

    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager


def get_db_manager(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> DatabaseManager:
    """
    Get DatabaseManager for a specific project.

    Args:
        project_id: Project identifier
        project_service: Injected ProjectService

    Returns:
        DatabaseManager instance

    Raises:
        HTTPException: If project not found
    """
    try:
        return project_service.get_db_manager(project_id)
    except Exception as e:
        logger.error(f"Error getting database manager for project {project_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


def get_vector_manager(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> VectorStoreManager:
    """
    Get VectorStoreManager for a specific project.

    Args:
        project_id: Project identifier
        project_service: Injected ProjectService

    Returns:
        VectorStoreManager instance

    Raises:
        HTTPException: If project not found
    """
    global _vector_managers

    # Return cached instance if available
    if project_id in _vector_managers:
        return _vector_managers[project_id]

    # Create new instance and cache it
    try:
        vector_manager = project_service.get_vector_manager(project_id)
        _vector_managers[project_id] = vector_manager
        return vector_manager
    except Exception as e:
        logger.error(f"Error getting vector manager for project {project_id}: {e}")
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")


def get_paper_service(
    project_id: str,
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)],
    vector_manager: Annotated[VectorStoreManager, Depends(get_vector_manager)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
    project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> PaperService:
    """
    Get PaperService for a specific project.

    Args:
        project_id: Project identifier
        db_manager: Injected DatabaseManager
        vector_manager: Injected VectorStoreManager
        llm_manager: Injected LLMManager
        prompt_manager: Injected PromptManager
        project_service: Injected ProjectService

    Returns:
        PaperService instance
    """
    project_root = project_service.projects_root / project_id
    return PaperService(
        db_manager=db_manager,
        vector_manager=vector_manager,
        llm_manager=llm_manager,
        prompt_manager=prompt_manager,
        project_root=project_root
    )


def get_chat_service(
    project_id: str,
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)],
    vector_manager: Annotated[VectorStoreManager, Depends(get_vector_manager)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)],
    project_service: Annotated[ProjectService, Depends(get_project_service)]
) -> ChatService:
    """
    Get ChatService for a specific project.

    Args:
        project_id: Project identifier
        db_manager: Injected DatabaseManager
        vector_manager: Injected VectorStoreManager
        llm_manager: Injected LLMManager
        prompt_manager: Injected PromptManager
        project_service: Injected ProjectService

    Returns:
        ChatService instance
    """
    project_root = project_service.projects_root / project_id
    return ChatService(
        db_manager=db_manager,
        vector_manager=vector_manager,
        llm_manager=llm_manager,
        prompt_manager=prompt_manager,
        project_root=project_root
    )


def get_recommendation_service(
    project_id: str,
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)],
    vector_manager: Annotated[VectorStoreManager, Depends(get_vector_manager)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)]
) -> RecommendationService:
    """
    Get RecommendationService for a specific project.

    Args:
        project_id: Project identifier
        db_manager: Injected DatabaseManager
        vector_manager: Injected VectorStoreManager
        llm_manager: Injected LLMManager
        prompt_manager: Injected PromptManager

    Returns:
        RecommendationService instance
    """
    return RecommendationService(
        db_manager=db_manager,
        vector_manager=vector_manager,
        llm_manager=llm_manager,
        prompt_manager=prompt_manager
    )


def get_settings_service() -> SettingsService:
    """
    Get SettingsService instance.

    Returns:
        SettingsService instance
    """
    return SettingsService()


def get_queue_service(
    db_manager: Annotated[DatabaseManager, Depends(get_db_manager)]
) -> QueueService:
    """
    Get QueueService for a specific project.

    Args:
        db_manager: Injected DatabaseManager

    Returns:
        QueueService instance
    """
    return QueueService(db_manager=db_manager)
