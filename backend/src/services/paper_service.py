"""
Paper management service.

Handles paper upload, processing, and CRUD operations.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, BinaryIO
from datetime import datetime
import shutil

from src.core.database import DatabaseManager, Paper
from src.core.vector_store import VectorStoreManager
from src.core.llm import LLMManager, PromptManager
from src.core.utils import PDFProcessor, DirectoryManager
from src.modules.file_manager.processor import PaperProcessor
from src.models.schemas import PaperResponse, PaperUpdate
from src.config import get_settings

logger = logging.getLogger(__name__)


class PaperService:
    """Service for managing papers."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        vector_manager: VectorStoreManager,
        llm_manager: LLMManager,
        prompt_manager: PromptManager,
        project_root: Path
    ):
        """
        Initialize paper service.

        Args:
            db_manager: Database manager instance
            vector_manager: Vector store manager instance
            llm_manager: LLM manager instance
            prompt_manager: Prompt manager instance
            project_root: Root directory of the project
        """
        self.db_manager = db_manager
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.prompt_manager = prompt_manager
        self.project_root = project_root

        # Get settings for PDF processing configuration
        settings = get_settings()

        # Initialize directory manager and PDF processor
        # All paper-related directories should be inside data/
        self.directory_manager = DirectoryManager(base_path=project_root / "data")
        self.pdf_processor = PDFProcessor(
            enable_llm_cleaning=settings.enable_llm_pdf_cleaning,
            llm_manager=llm_manager,
            pages_per_chunk=settings.pdf_pages_per_chunk,
            enable_reference_detection=True,
            auto_skip_threshold=100,
            enable_parallel=settings.enable_parallel_pdf_processing,
            max_concurrent=settings.max_concurrent_pdf_chunks
        )

        # Initialize paper processor
        self.paper_processor = PaperProcessor(
            db_manager=db_manager,
            vector_manager=vector_manager,
            llm_manager=llm_manager,
            prompt_manager=prompt_manager,
            directory_manager=self.directory_manager,
            pdf_processor=self.pdf_processor
        )

        logger.info("Initialized PaperService")

    def get_papers(
        self,
        status: Optional[str] = None,
        year: Optional[int] = None,
        author: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PaperResponse]:
        """
        Get papers with optional filters.

        Args:
            status: Filter by reading status
            year: Filter by publication year
            author: Filter by author name
            limit: Maximum number of results

        Returns:
            List of PaperResponse objects
        """
        if status:
            papers = self.db_manager.get_papers_by_status(status)
        elif year:
            papers = self.db_manager.get_papers_by_year_range(year, year)
        elif author:
            papers = self.db_manager.search_papers_by_author(author)
        else:
            papers = self.db_manager.get_all_papers(limit=limit)

        return [self._paper_to_response(paper) for paper in papers]

    def get_paper(self, paper_id: int) -> Optional[PaperResponse]:
        """
        Get a specific paper.

        Args:
            paper_id: Paper ID

        Returns:
            PaperResponse or None
        """
        paper = self.db_manager.get_paper_by_id(paper_id)
        if not paper:
            return None
        return self._paper_to_response(paper)

    async def upload_paper(
        self,
        file: BinaryIO,
        filename: str
    ) -> PaperResponse:
        """
        Upload and process a new paper.

        Args:
            file: File object
            filename: Original filename

        Returns:
            Created PaperResponse
        """
        # Save file to _NEW_PAPERS directory
        new_papers_dir = self.project_root / "data" / "_NEW_PAPERS"
        new_papers_dir.mkdir(parents=True, exist_ok=True)

        file_path = new_papers_dir / filename

        # Write file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file, f)

        logger.info(f"Uploaded paper: {filename}")

        # Process the paper (this will be done in background in production)
        try:
            processed_paper = await self.paper_processor.process_new_paper(file_path, force_clean=False)
            logger.info(f"Successfully processed paper: {filename}")
        except Exception as e:
            logger.error(f"Error processing paper {filename}: {e}")
            raise

        # Get the processed paper from database
        paper = self.db_manager.get_paper_by_original_filename(filename)
        if not paper:
            raise ValueError(f"Paper {filename} not found in database after processing")

        return self._paper_to_response(paper)

    def update_paper(self, paper_id: int, updates: PaperUpdate) -> Optional[PaperResponse]:
        """
        Update paper metadata.

        Args:
            paper_id: Paper ID
            updates: PaperUpdate object

        Returns:
            Updated PaperResponse or None
        """
        update_dict = {}

        if updates.status is not None:
            update_dict['status'] = updates.status
        if updates.tags is not None:
            self.db_manager.update_paper_tags(paper_id, updates.tags)
        if updates.summary is not None:
            update_dict['summary'] = updates.summary

        if update_dict:
            success = self.db_manager.update_paper(paper_id, update_dict)
            if not success:
                return None

        paper = self.db_manager.get_paper_by_id(paper_id)
        if not paper:
            return None

        return self._paper_to_response(paper)

    def delete_paper(self, paper_id: int) -> bool:
        """
        Delete a paper.

        Args:
            paper_id: Paper ID

        Returns:
            True if deleted, False otherwise
        """
        paper = self.db_manager.get_paper_by_id(paper_id)
        if not paper:
            return False

        # Delete vector embeddings
        try:
            # Delete all chunks for this paper
            chunk_ids = [f"paper_{paper_id}_chunk_{i}" for i in range(1000)]  # Assume max 1000 chunks
            self.vector_manager.collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.warning(f"Error deleting vector embeddings for paper {paper_id}: {e}")

        # Delete from database
        success = self.db_manager.delete_paper(paper_id)

        # Optionally delete PDF file (move to _FAILED or delete permanently)
        if success and paper.file_path:
            try:
                file_path = Path(paper.file_path)
                if file_path.exists():
                    # Move to _FAILED directory instead of deleting
                    failed_dir = self.project_root / "data" / "_FAILED"
                    failed_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(file_path), str(failed_dir / file_path.name))
            except Exception as e:
                logger.warning(f"Error moving PDF file for paper {paper_id}: {e}")

        return success

    async def reprocess_paper(self, paper_id: int) -> PaperResponse:
        """
        Reprocess a paper (re-run PDF extraction and embedding).

        Args:
            paper_id: Paper ID

        Returns:
            Updated PaperResponse
        """
        paper = self.db_manager.get_paper_by_id(paper_id)
        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        file_path = Path(paper.file_path)
        if not file_path.exists():
            raise ValueError(f"PDF file not found: {paper.file_path}")

        # Delete existing vector embeddings
        try:
            chunk_ids = [f"paper_{paper_id}_chunk_{i}" for i in range(1000)]
            self.vector_manager.collection.delete(ids=chunk_ids)
        except Exception as e:
            logger.warning(f"Error deleting old embeddings: {e}")

        # Reprocess
        processed_paper = await self.paper_processor.process_new_paper(file_path, force_clean=False)

        # Get updated paper
        paper = self.db_manager.get_paper_by_id(paper_id)
        return self._paper_to_response(paper)

    async def analyze_paper(
        self,
        file: BinaryIO,
        filename: str
    ) -> PaperResponse:
        """
        Phase 2C: Analyze uploaded PDF and add to queue workflow.

        Papers go through: detected → analyzed → (user selects) → queued → processing

        Args:
            file: File object
            filename: Original filename

        Returns:
            PaperResponse with status='analyzed' and metadata for user review
        """
        import fitz  # PyMuPDF

        # Save file to temp location
        new_papers_dir = self.project_root / "data" / "_NEW_PAPERS"
        new_papers_dir.mkdir(parents=True, exist_ok=True)
        file_path = new_papers_dir / filename

        # Write file
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file, f)

        logger.info(f"Analyzing paper for queue: {filename}")

        try:
            # Quick analysis: extract basic metadata
            from src.core.utils.pdf_cleaner import PDFBlockExtractor, ReferenceSectionDetector
            import fitz

            # Get page count and basic info
            extractor = PDFBlockExtractor()
            pages_data = extractor.process_pdf(file_path)
            original_page_count = len(pages_data)

            # Detect references to get accurate page count
            detector = ReferenceSectionDetector()
            cutoff_page = detector.detect_cutoff_page(pages_data, self.llm_manager)

            if cutoff_page:
                pages_data_after_refs = detector.truncate_pages(pages_data, cutoff_page)
                page_count_after_refs = len(pages_data_after_refs)
                references_removed = True
            else:
                page_count_after_refs = original_page_count
                references_removed = False

            # Extract title from PDF metadata or filename
            pdf_doc = fitz.open(file_path)
            pdf_metadata = pdf_doc.metadata or {}
            title = pdf_metadata.get('title', '') or filename.replace('.pdf', '').replace('_', ' ')
            pdf_doc.close()

            # Estimate processing time/cost for user decision
            threshold = self.pdf_processor.auto_skip_threshold or 100
            needs_confirmation = page_count_after_refs > threshold
            estimated_time_minutes = (page_count_after_refs * 8) / 60  # 8 seconds per page
            estimated_cost_usd = page_count_after_refs * 0.001  # $0.001 per page

            # Create paper record with 'analyzed' status for queue
            paper_data = {
                'filename': filename,
                'original_filename': filename,
                'title': title,
                'authors': '',
                'year': None,
                'summary': None,
                'abstract': None,
                'status': 'analyzed',  # Ready for user selection in Pending tab
                'tags': '',
                'file_path': str(file_path),
                'page_count': original_page_count
            }

            paper = self.db_manager.add_paper(paper_data)

            # Build response with metadata for user decision
            response = self._paper_to_response(paper)
            response.confirmation_metadata = {
                'page_count': original_page_count,
                'page_count_after_refs': page_count_after_refs,
                'estimated_time_minutes': round(estimated_time_minutes, 1),
                'estimated_cost_usd': round(estimated_cost_usd, 2),
                'references_removed': references_removed,
                'threshold': threshold,
                'needs_confirmation': needs_confirmation
            }

            logger.info(f"Paper {filename} added to queue with 'analyzed' status ({page_count_after_refs} pages)")
            return response

        except Exception as e:
            logger.error(f"Error analyzing paper {filename}: {e}")
            raise

    async def confirm_paper_processing(
        self,
        paper_id: int,
        force_clean: bool
    ) -> PaperResponse:
        """
        Complete processing after user confirmation.

        Args:
            paper_id: Paper ID
            force_clean: Whether to force clean despite threshold

        Returns:
            Processed PaperResponse
        """
        paper = self.db_manager.get_paper_by_id(paper_id)
        if not paper:
            raise ValueError(f"Paper {paper_id} not found")

        if paper.status != 'pending_confirmation':
            raise ValueError(f"Paper {paper_id} is not pending confirmation (status: {paper.status})")

        file_path = Path(paper.file_path)
        if not file_path.exists():
            raise ValueError(f"PDF file not found: {paper.file_path}")

        logger.info(f"Confirming processing for paper {paper_id}: force_clean={force_clean}")

        # Update processor threshold temporarily for this paper
        original_threshold = self.pdf_processor.auto_skip_threshold

        try:
            if force_clean:
                # User wants to force clean - we need to pass force_clean to the processor
                # For now, we'll temporarily disable the threshold
                logger.info("Force clean requested - disabling threshold for this paper")
                self.pdf_processor.auto_skip_threshold = None
            else:
                # User wants to skip cleaning - keep threshold as is
                logger.info("Skip cleaning requested - keeping threshold")

            # Process the paper
            processed_paper = await self.paper_processor.process_new_paper(file_path, force_clean=force_clean)

            # Get the processed paper
            paper = self.db_manager.get_paper_by_id(paper_id)
            if not paper:
                raise ValueError(f"Paper {paper_id} not found after processing")

            return self._paper_to_response(paper)

        finally:
            # Restore original threshold
            self.pdf_processor.auto_skip_threshold = original_threshold

    def _paper_to_response(self, paper: Paper) -> PaperResponse:
        """
        Convert Paper model to PaperResponse.

        Args:
            paper: Paper database model

        Returns:
            PaperResponse object
        """
        return PaperResponse(
            id=paper.id,
            filename=paper.filename,
            original_filename=paper.original_filename,
            title=paper.title,
            authors=paper.get_authors_list(),
            year=paper.year,
            summary=paper.summary,
            abstract=paper.abstract,
            status=paper.status,
            tags=paper.get_tags_list(),
            date_added=paper.date_added,
            file_path=paper.file_path,
            doi=paper.doi,
            journal=paper.journal,
            page_count=paper.page_count,
            citation_count=paper.citation_count,
            venue=paper.venue,
            semantic_scholar_url=paper.semantic_scholar_url
        )
