"""Recommendations API endpoints - handles paper recommendations."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from src.models.schemas import RecommendationQueryRequest, RecommendationResponse
from src.services.recommendation_service import RecommendationService
from src.core.database import DatabaseManager
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.dependencies import get_db_manager, get_vector_manager, get_llm_manager, get_prompt_manager, get_project_service
from src.services import ProjectService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=RecommendationResponse)
async def get_recommendations_by_query(
    request: RecommendationQueryRequest,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)]
):
    """
    Get paper recommendations based on a text query.

    This endpoint uses AI-powered analysis to:
    - Identify research gaps in your current literature
    - Search Semantic Scholar for relevant papers
    - Filter out papers you already have
    - Provide targeted recommendations with reasoning

    Args:
        request: Recommendation request with query and parameters

    Returns:
        RecommendationResponse with list of recommended papers and gap analysis

    Raises:
        HTTPException: If recommendation generation fails
    """
    try:
        logger.info(f"Getting recommendations for project {request.project_id}, query: {request.query}")

        # Get database and vector managers for the project
        db_manager = project_service.get_db_manager(request.project_id)
        vector_manager = project_service.get_vector_manager(request.project_id)

        # Create recommendation service with project-specific managers
        recommendation_service = RecommendationService(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager
        )

        response = await recommendation_service.get_recommendations(request)

        logger.info(f"Generated {len(response.recommendations)} recommendations")
        return response

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )
