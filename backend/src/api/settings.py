"""Settings API endpoints - handles application configuration."""

import logging
from typing import List, Annotated
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from src.models.schemas import (
    EmbeddingModelInfo,
    LLMSettingsResponse
)
from src.services.settings_service import SettingsService
from src.dependencies import get_settings_service
from src.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/embedding-models", response_model=List[EmbeddingModelInfo])
async def get_embedding_models(
    settings_service: Annotated[SettingsService, Depends(get_settings_service)] = None
):
    """
    Get available embedding models.

    Returns:
        List of available embedding models with their specifications

    """
    logger.info("Fetching available embedding models")
    return settings_service.get_embedding_models()


@router.get("/llm", response_model=LLMSettingsResponse)
async def get_llm_settings(
    settings_service: Annotated[SettingsService, Depends(get_settings_service)] = None
):
    """
    Get LLM provider settings.

    Returns:
        Current LLM configuration including providers and models
    """
    logger.info("Fetching LLM settings")
    return settings_service.get_llm_settings()
