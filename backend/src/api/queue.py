"""
Queue API Router

Endpoints for managing the paper processing queue.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Annotated
import logging

from src.dependencies import get_queue_service, get_project_service, get_paper_service
from src.services.queue_service import QueueService
from src.services.project_service import ProjectService
from src.services.paper_service import PaperService
from src.models.schemas import (
    PaperResponse,
    QueueStatusResponse,
    BatchSelectRequest,
    MessageResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/queue", tags=["queue"])


@router.get("/pending", response_model=List[PaperResponse])
async def list_pending_papers(
    project_id: str,
    queue_service: QueueService = Depends(get_queue_service),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get list of papers awaiting user selection (status='analyzed').

    Args:
        project_id: Project ID (string identifier like 'test_proj')

    Returns:
        List of papers with status='analyzed'
    """
    # Verify project exists
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get papers with status='analyzed' directly from database
    db_manager = project_service.get_db_manager(project_id)
    with db_manager.get_session() as session:
        from sqlmodel import select
        from src.core.database import Paper as PaperModel
        statement = select(PaperModel).where(PaperModel.status == "analyzed")
        papers = session.exec(statement).all()

        # Convert Paper models to PaperResponse with proper list parsing
        responses = []
        for paper in papers:
            paper_dict = paper.model_dump()
            # Parse JSON string fields to lists
            paper_dict['authors'] = paper.get_authors_list()
            paper_dict['tags'] = paper.get_tags_list()
            responses.append(PaperResponse(**paper_dict))

    return responses


@router.get("/queued", response_model=List[PaperResponse])
async def list_queued_papers(
    project_id: int,
    queue_service: QueueService = Depends(get_queue_service),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get list of papers in processing queue (status='queued').

    Args:
        project_id: Project ID

    Returns:
        List of papers with status='queued'
    """
    # Verify project exists
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    papers = queue_service.list_queued_papers(project_id)
    return [PaperResponse.model_validate(paper) for paper in papers]


@router.post("/select", response_model=MessageResponse)
async def select_papers_for_processing(
    request: BatchSelectRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Mark selected papers for processing and start FULL processing immediately.

    This does complete paper processing: text extraction, metadata, embeddings, AI summary.

    Args:
        request: BatchSelectRequest with project_id and paper_ids

    Returns:
        Success message with count
    """
    # Verify project exists
    project = project_service.get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get services for this specific project - manual instantiation to avoid dependency injection issues
    from src.dependencies import (
        get_llm_manager,
        get_prompt_manager
    )

    db_manager = project_service.get_db_manager(request.project_id)
    vector_manager = project_service.get_vector_manager(request.project_id)
    llm_manager = get_llm_manager()
    prompt_manager = get_prompt_manager()
    project_root = project_service.projects_root / request.project_id

    paper_service = PaperService(
        db_manager=db_manager,
        vector_manager=vector_manager,
        llm_manager=llm_manager,
        prompt_manager=prompt_manager,
        project_root=project_root
    )
    queue_service = QueueService(db_manager=db_manager)

    # Mark papers as queued
    queued_count = queue_service.select_papers_for_processing(
        request.project_id,
        request.paper_ids
    )

    # Start FULL background processing
    import asyncio
    from pathlib import Path
    from src.core.database import Paper as PaperModel
    from sqlmodel import select

    async def process_queued_papers():
        """Background task to FULLY process papers with WebSocket progress updates."""
        from src.api.websocket import manager

        for paper_id in request.paper_ids:
            try:
                # Update status to processing
                queue_service.update_paper_status(paper_id, "processing")
                logger.info(f"🔄 Starting FULL processing for paper {paper_id}")

                # Get paper from database
                with db_manager.get_session() as session:
                    statement = select(PaperModel).where(PaperModel.id == paper_id)
                    paper = session.exec(statement).first()

                    if not paper:
                        logger.warning(f"Paper {paper_id} not found")
                        queue_service.update_paper_status(paper_id, "analyzed")
                        continue

                    pdf_path = Path(paper.file_path)
                    if not pdf_path.exists():
                        logger.error(f"PDF file not found: {pdf_path}")
                        queue_service.update_paper_status(paper_id, "analyzed")
                        continue

                    paper_title = paper.title or paper.filename

                # Broadcast: Starting (0%)
                await manager.broadcast_to_project(request.project_id, {
                    "type": "processing_progress",
                    "paper_id": paper_id,
                    "title": paper_title,
                    "progress": 0,
                    "step": "Starting processing..."
                })

                # Delete the analyzed record first (to avoid duplicate detection)
                with db_manager.get_session() as session:
                    session.delete(paper)
                    session.commit()
                    logger.info(f"Deleted analyzed record for paper {paper_id} before reprocessing")

                # Define real progress callback that broadcasts actual processing stages
                async def progress_callback(progress: int, step: str):
                    """Send REAL progress updates from actual processing stages."""
                    await manager.broadcast_to_project(request.project_id, {
                        "type": "processing_progress",
                        "paper_id": paper_id,
                        "title": paper_title,
                        "progress": progress,
                        "step": step
                    })

                # Process the paper with REAL progress tracking:
                # - Text extraction (25%)
                # - Metadata generation (40%)
                # - AI summary (60%)
                # - Embeddings (80%)
                # - Semantic Scholar enrichment (included in earlier stages)
                processed_paper = await paper_service.paper_processor.process_new_paper(
                    pdf_path,
                    force_clean=False,
                    progress_callback=progress_callback
                )

                if processed_paper:
                    logger.info(f"✅ Successfully FULLY processed paper (new ID: {processed_paper.id}): {processed_paper.title}")

                    # Broadcast: Complete (100%)
                    await manager.broadcast_to_project(request.project_id, {
                        "type": "processing_complete",
                        "paper_id": paper_id,
                        "new_paper_id": processed_paper.id,
                        "title": processed_paper.title,
                        "progress": 100
                    })
                else:
                    logger.error(f"❌ Failed to process paper from {pdf_path}")

                    # Broadcast: Failed
                    await manager.broadcast_to_project(request.project_id, {
                        "type": "processing_failed",
                        "paper_id": paper_id,
                        "title": paper_title,
                        "error": "Processing failed"
                    })

            except Exception as e:
                logger.error(f"Error processing paper {paper_id}: {e}", exc_info=True)

                # Broadcast: Error
                await manager.broadcast_to_project(request.project_id, {
                    "type": "processing_failed",
                    "paper_id": paper_id,
                    "error": str(e)
                })

    # Start processing in background
    asyncio.create_task(process_queued_papers())

    return MessageResponse(
        message=f"Successfully started FULL processing of {queued_count}/{len(request.paper_ids)} papers"
    )


@router.post("/reject", response_model=MessageResponse)
async def reject_papers(
    request: BatchSelectRequest,
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Reject selected papers (delete from queue).

    Args:
        request: BatchSelectRequest with project_id and paper_ids

    Returns:
        Success message with count
    """
    # Verify project exists
    project = project_service.get_project(request.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Get DB manager and queue service for this specific project
    db_manager = project_service.get_db_manager(request.project_id)
    queue_service = QueueService(db_manager=db_manager)

    rejected_count = queue_service.reject_papers(
        request.project_id,
        request.paper_ids
    )

    return MessageResponse(
        message=f"Successfully rejected {rejected_count}/{len(request.paper_ids)} papers"
    )


@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status(
    project_id: str,
    queue_service: QueueService = Depends(get_queue_service),
    project_service: ProjectService = Depends(get_project_service)
):
    """
    Get current queue processing status.

    Args:
        project_id: Project ID (string identifier)

    Returns:
        Queue status with counts and current processing paper
    """
    # Verify project exists
    project = project_service.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    status = queue_service.get_queue_status(project_id)
    return QueueStatusResponse(**status)
