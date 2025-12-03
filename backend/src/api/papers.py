"""Papers API endpoints - handles paper management and PDF uploads."""

import logging
from typing import List, Optional, Annotated
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status
from fastapi.responses import FileResponse

from src.models.schemas import PaperResponse, PaperUpdate
from src.services.paper_service import PaperService
from src.dependencies import get_paper_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/", response_model=List[PaperResponse])
async def list_papers(
    project_id: str = Query(..., description="Project ID"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by reading status"),
    year: Optional[int] = Query(None, description="Filter by publication year"),
    author: Optional[str] = Query(None, description="Filter by author name"),
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Maximum number of results"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    List all papers in the project with optional filters.

    Args:
        project_id: Project identifier
        status_filter: Filter by reading status (unread, reading, read)
        year: Filter by publication year
        author: Filter by author name (partial match)
        limit: Maximum number of results

    Returns:
        List of papers matching the criteria
    """
    try:
        papers = paper_service.get_papers(
            status=status_filter,
            year=year,
            author=author,
            limit=limit
        )
        logger.info(f"Listed {len(papers)} papers for project {project_id}")
        return papers
    except Exception as e:
        logger.error(f"Error listing papers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving papers: {str(e)}"
        )


@router.post("/upload", response_model=PaperResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    project_id: str = Query(..., description="Project ID"),
    file: UploadFile = File(..., description="PDF file to upload"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Upload a new paper PDF for processing.

    The paper will be processed automatically:
    - PDF text extraction
    - Metadata extraction (title, authors, year, abstract)
    - Vector embeddings generation
    - Storage in database and vector store

    Args:
        project_id: Project identifier
        file: PDF file to upload

    Returns:
        Created paper with extracted metadata

    Raises:
        HTTPException: If file is not a PDF or processing fails
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    try:
        logger.info(f"Uploading paper: {file.filename} to project {project_id}")

        # Upload and process paper
        paper = await paper_service.upload_paper(file.file, file.filename)

        logger.info(f"Successfully uploaded and processed paper: {file.filename}")
        return paper

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error uploading paper: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing paper: {str(e)}"
        )


@router.post("/upload-batch", response_model=List[PaperResponse], status_code=status.HTTP_201_CREATED)
async def upload_batch(
    project_id: str = Query(..., description="Project ID"),
    files: List[UploadFile] = File(..., description="PDF files to upload"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Upload multiple paper PDFs for batch analysis.

    Each paper will be analyzed (quick scan without full processing):
    - PDF page count
    - Basic validation
    - Stored in pending queue for user review

    Args:
        project_id: Project identifier
        files: List of PDF files to upload

    Returns:
        List of analyzed papers awaiting user confirmation

    Raises:
        HTTPException: If files are not PDFs or analysis fails
    """
    # Validate all files are PDFs
    non_pdf_files = [f.filename for f in files if not f.filename.endswith('.pdf')]
    if non_pdf_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are supported. Invalid files: {', '.join(non_pdf_files)}"
        )

    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    results = []
    errors = []

    logger.info(f"Batch uploading {len(files)} papers to project {project_id}")

    # Process each file sequentially
    for idx, file in enumerate(files, 1):
        try:
            logger.info(f"Analyzing paper {idx}/{len(files)}: {file.filename}")

            # Analyze paper (quick operation, no full processing)
            paper = await paper_service.analyze_paper(file.file, file.filename)
            results.append(paper)

            logger.info(f"Successfully analyzed paper {idx}/{len(files)}: {file.filename}")

        except Exception as e:
            error_msg = f"{file.filename}: {str(e)}"
            logger.error(f"Error analyzing paper {file.filename}: {e}")
            errors.append(error_msg)
            # Continue with other files instead of failing entire batch

    # If all files failed, return error
    if len(errors) == len(files):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"All uploads failed: {'; '.join(errors)}"
        )

    # If some files failed, log warning but return successful ones
    if errors:
        logger.warning(f"Some uploads failed ({len(errors)}/{len(files)}): {'; '.join(errors)}")

    logger.info(f"Batch upload complete: {len(results)}/{len(files)} papers analyzed successfully")
    return results


@router.post("/analyze-upload", response_model=PaperResponse, status_code=status.HTTP_200_OK)
async def analyze_upload(
    project_id: str = Query(..., description="Project ID"),
    file: UploadFile = File(..., description="PDF file to analyze"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Analyze uploaded PDF without full processing.

    Returns confirmation requirement if PDF >100 pages (after reference removal).
    For small PDFs, processes immediately.

    Args:
        project_id: Project identifier
        file: PDF file to analyze

    Returns:
        PaperResponse with needs_confirmation flag if large,
        or fully processed paper if small

    Raises:
        HTTPException: If file is not a PDF or analysis fails
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported"
        )

    try:
        logger.info(f"Analyzing paper: {file.filename} for project {project_id}")

        # Analyze paper
        paper = await paper_service.analyze_paper(file.file, file.filename)

        if paper.needs_confirmation:
            logger.info(f"Paper {file.filename} needs confirmation")
        else:
            logger.info(f"Paper {file.filename} processed successfully (small PDF)")

        return paper

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error analyzing paper: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing paper: {str(e)}"
        )


@router.post("/{paper_id}/confirm-processing", response_model=PaperResponse)
async def confirm_processing(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    force_clean: bool = Query(..., description="Force LLM cleaning despite threshold"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Complete processing after user confirmation for large PDFs.

    Args:
        paper_id: Paper ID (from analyze_upload response)
        project_id: Project identifier
        force_clean: True = clean anyway, False = skip cleaning

    Returns:
        Fully processed paper

    Raises:
        HTTPException: If paper not found or processing fails
    """
    try:
        logger.info(f"Confirming processing for paper {paper_id}: force_clean={force_clean}")

        paper = await paper_service.confirm_paper_processing(paper_id, force_clean)

        logger.info(f"Successfully processed paper {paper_id}")
        return paper

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error confirming paper processing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing paper: {str(e)}"
        )


@router.get("/{paper_id}", response_model=PaperResponse)
async def get_paper(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Get details of a specific paper.

    Args:
        paper_id: Paper ID
        project_id: Project identifier

    Returns:
        Paper details with all metadata

    Raises:
        HTTPException: If paper not found
    """
    paper = paper_service.get_paper(paper_id)

    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )

    return paper


@router.get("/{paper_id}/pdf")
async def download_paper_pdf(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Download the PDF file for a paper.

    Args:
        paper_id: Paper ID
        project_id: Project identifier

    Returns:
        PDF file

    Raises:
        HTTPException: If paper or file not found
    """
    paper = paper_service.get_paper(paper_id)

    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )

    file_path = Path(paper.file_path)
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file not found for paper {paper_id}"
        )

    return FileResponse(
        path=file_path,
        media_type="application/pdf",
        filename=paper.original_filename
    )


@router.get("/{paper_id}/summary-pdf")
async def download_paper_summary_pdf(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Download the AI-generated detailed summary PDF for a paper.

    Args:
        paper_id: Paper ID
        project_id: Project identifier

    Returns:
        Summary PDF file

    Raises:
        HTTPException: If paper or summary file not found
    """
    paper = paper_service.get_paper(paper_id)

    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper {paper_id} not found"
        )

    # Construct summary PDF path
    file_path = Path(paper.file_path)
    summaries_dir = file_path.parent / "summaries"
    base_name = file_path.stem
    summary_filename = f"{base_name}_summary.pdf"
    summary_path = summaries_dir / summary_filename

    if not summary_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Summary PDF not found for paper {paper_id}"
        )

    return FileResponse(
        path=summary_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={summary_filename}"}
    )


@router.put("/{paper_id}", response_model=PaperResponse)
async def update_paper(
    paper_id: int,
    updates: PaperUpdate,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Update paper metadata.

    Args:
        paper_id: Paper ID
        updates: Fields to update (status, tags, summary)
        project_id: Project identifier

    Returns:
        Updated paper

    Raises:
        HTTPException: If paper not found
    """
    try:
        paper = paper_service.update_paper(paper_id, updates)

        if not paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper {paper_id} not found"
            )

        logger.info(f"Updated paper {paper_id}")
        return paper

    except Exception as e:
        logger.error(f"Error updating paper {paper_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating paper: {str(e)}"
        )


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_paper(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Delete a paper.

    This will:
    - Remove paper from database
    - Delete vector embeddings
    - Move PDF file to _FAILED directory

    Args:
        paper_id: Paper ID
        project_id: Project identifier

    Raises:
        HTTPException: If paper not found
    """
    try:
        success = paper_service.delete_paper(paper_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper {paper_id} not found"
            )

        logger.info(f"Deleted paper {paper_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting paper {paper_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting paper: {str(e)}"
        )


@router.post("/{paper_id}/reprocess", response_model=PaperResponse)
async def reprocess_paper(
    paper_id: int,
    project_id: str = Query(..., description="Project ID"),
    paper_service: Annotated[PaperService, Depends(get_paper_service)] = None
):
    """
    Reprocess a paper.

    This will re-run:
    - PDF text extraction
    - Metadata extraction
    - Vector embeddings generation

    Useful if processing failed or you want to use updated extraction logic.

    Args:
        paper_id: Paper ID
        project_id: Project identifier

    Returns:
        Reprocessed paper with updated metadata

    Raises:
        HTTPException: If paper not found or PDF file missing
    """
    try:
        paper = await paper_service.reprocess_paper(paper_id)
        logger.info(f"Reprocessed paper {paper_id}")
        return paper

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error reprocessing paper {paper_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing paper: {str(e)}"
        )
