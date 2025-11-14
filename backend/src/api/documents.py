"""Documents API endpoints - handles document analysis and listing."""

import logging
import os
from pathlib import Path
from typing import Annotated, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from src.models.schemas import (
    DocumentInfo,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    DocumentRecommendationRequest,
    RecommendationResponse,
    PaperRecommendationItem
)
from src.modules.document_processor import DocumentProcessor, DocumentAnalysis
from src.services.recommendation_service import RecommendationService
from src.core.llm import LLMManager, PromptManager
from src.dependencies import (
    get_project_service,
    get_llm_manager,
    get_prompt_manager
)
from src.services import ProjectService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/{project_id}/list", response_model=List[DocumentInfo])
async def list_documents(
    project_id: str,
    project_service: Annotated[ProjectService, Depends(get_project_service)]
):
    """
    List all .docx documents in the project's documents folder.

    Args:
        project_id: Project identifier

    Returns:
        List of document information objects
    """
    try:
        logger.info(f"Listing documents for project {project_id}")

        # Get project path
        project = project_service.get_project(project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {project_id} not found"
            )

        # Get documents folder path
        documents_folder = Path(project.path) / "documents"

        if not documents_folder.exists():
            logger.info(f"Documents folder does not exist: {documents_folder}")
            return []

        # List all .docx files
        documents = []
        for docx_file in documents_folder.glob("*.docx"):
            # Skip temp files
            if docx_file.name.startswith("~$"):
                continue

            stat = docx_file.stat()
            documents.append(DocumentInfo(
                filename=docx_file.name,
                file_path=str(docx_file.relative_to(project.path)),
                size_bytes=stat.st_size,
                last_modified=datetime.fromtimestamp(stat.st_mtime)
            ))

        logger.info(f"Found {len(documents)} documents")
        return documents

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.post("/analyze", response_model=List[DocumentAnalysisResponse])
async def analyze_documents(
    request: DocumentAnalysisRequest,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)]
):
    """
    Analyze one or more documents to extract research content.

    This endpoint:
    - Extracts text from .docx files
    - Identifies research questions and key concepts
    - Finds literature gaps
    - Determines document type

    Args:
        request: Document analysis request with project ID and document paths

    Returns:
        List of document analysis results

    Raises:
        HTTPException: If analysis fails
    """
    try:
        logger.info(f"Analyzing {len(request.document_paths)} documents for project {request.project_id}")

        # Get project
        project = project_service.get_project(request.project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {request.project_id} not found"
            )

        # Get managers for the project
        db_manager = project_service.get_db_manager(request.project_id)
        vector_manager = project_service.get_vector_manager(request.project_id)

        # Create document processor
        doc_processor = DocumentProcessor(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager
        )

        # Analyze each document
        analyses = []
        for doc_path in request.document_paths:
            # Construct full path
            full_path = Path(project.path) / doc_path

            if not full_path.exists():
                logger.warning(f"Document not found: {full_path}")
                continue

            # Analyze document
            analysis = doc_processor.analyze_document(full_path)

            # Convert to response model
            analyses.append(DocumentAnalysisResponse(
                title=analysis.title,
                word_count=analysis.word_count,
                research_questions=analysis.research_questions,
                key_concepts=analysis.key_concepts,
                literature_gaps=analysis.literature_gaps,
                citations_present=analysis.citations_present,
                citations_needed=analysis.citations_needed,
                document_type=analysis.document_type,
                confidence_score=analysis.confidence_score
            ))

        logger.info(f"Successfully analyzed {len(analyses)} documents")
        return analyses

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing documents: {str(e)}"
        )


@router.post("/recommendations", response_model=RecommendationResponse)
async def get_document_based_recommendations(
    request: DocumentRecommendationRequest,
    project_service: Annotated[ProjectService, Depends(get_project_service)],
    llm_manager: Annotated[LLMManager, Depends(get_llm_manager)],
    prompt_manager: Annotated[PromptManager, Depends(get_prompt_manager)]
):
    """
    Get paper recommendations based on document analysis.

    This endpoint:
    - Analyzes selected .docx documents
    - Extracts key concepts and research questions
    - Identifies literature gaps
    - Generates targeted paper recommendations

    Args:
        request: Document recommendation request

    Returns:
        RecommendationResponse with recommendations based on document analysis

    Raises:
        HTTPException: If recommendation generation fails
    """
    try:
        logger.info(f"Generating document-based recommendations for project {request.project_id}")

        # Get project
        project = project_service.get_project(request.project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Project {request.project_id} not found"
            )

        # Get managers
        db_manager = project_service.get_db_manager(request.project_id)
        vector_manager = project_service.get_vector_manager(request.project_id)

        # Create document processor
        doc_processor = DocumentProcessor(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager
        )

        # Analyze all documents and combine insights
        all_concepts = []
        all_gaps = []
        all_questions = []

        for doc_path in request.document_paths:
            full_path = Path(project.path) / doc_path

            if not full_path.exists():
                continue

            analysis = doc_processor.analyze_document(full_path)
            all_concepts.extend(analysis.key_concepts)
            all_gaps.extend(analysis.literature_gaps)
            all_questions.extend(analysis.research_questions)

        # Create combined query from document analysis
        query_parts = []
        if all_questions:
            query_parts.append(f"Research questions: {', '.join(all_questions[:3])}")
        if all_concepts:
            query_parts.append(f"Key concepts: {', '.join(all_concepts[:5])}")
        if all_gaps:
            query_parts.append(f"Literature gaps: {', '.join(all_gaps[:3])}")

        combined_query = ". ".join(query_parts)

        if not combined_query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract meaningful content from documents"
            )

        logger.info(f"Generated query from documents: {combined_query[:200]}...")

        # Get recommendations using the combined query
        from src.models.schemas import RecommendationQueryRequest

        rec_request = RecommendationQueryRequest(
            project_id=request.project_id,
            query=combined_query,
            max_recommendations=request.max_recommendations
        )

        # Create recommendation service
        recommendation_service = RecommendationService(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager
        )

        response = await recommendation_service.get_recommendations(rec_request)

        logger.info(f"Generated {len(response.recommendations)} document-based recommendations")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating document-based recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )
