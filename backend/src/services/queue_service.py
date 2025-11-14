"""
Paper Queue Service

Manages the paper processing queue workflow:
1. File detection → status='detected'
2. Quick analysis → status='analyzed'
3. User selection → status='queued'
4. Processing → status='processing' → status='unread'
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlmodel import select

from src.core.database.operations import DatabaseManager
from src.core.database.models import Paper

logger = logging.getLogger(__name__)


class QueueService:
    """Service for managing paper processing queue."""

    def __init__(self, db_manager: DatabaseManager):
        """
        Initialize queue service.

        Args:
            db_manager: Database manager instance for a specific project
        """
        self.db = db_manager

    def list_pending_papers(self, project_id: str) -> List[Paper]:
        """
        Get list of papers awaiting user selection (status='analyzed').

        Args:
            project_id: Project ID (not used - each project has own DB)

        Returns:
            List of Paper objects with status='analyzed'
        """
        with self.db.get_session() as session:
            statement = select(Paper).where(
                Paper.status == "analyzed"
            ).order_by(Paper.date_added.desc())

            papers = session.exec(statement).all()
            return list(papers)

    def list_queued_papers(self, project_id: str) -> List[Paper]:
        """
        Get list of papers in processing queue (status='queued').

        Args:
            project_id: Project ID (not used - each project has own DB)

        Returns:
            List of Paper objects with status='queued'
        """
        with self.db.get_session() as session:
            statement = select(Paper).where(
                Paper.status == "queued"
            ).order_by(Paper.date_added.asc())

            papers = session.exec(statement).all()
            return list(papers)

    def get_processing_paper(self, project_id: str) -> Optional[Paper]:
        """
        Get currently processing paper (status='processing').

        Args:
            project_id: Project ID (not used - each project has own DB)

        Returns:
            Paper object or None
        """
        with self.db.get_session() as session:
            statement = select(Paper).where(
                Paper.status == "processing"
            ).limit(1)

            paper = session.exec(statement).first()
            return paper

    def select_papers_for_processing(self, project_id: str, paper_ids: List[int]) -> int:
        """
        Mark selected papers for processing (analyzed → queued).

        Args:
            project_id: Project ID (not used - each project has own DB)
            paper_ids: List of paper IDs to queue

        Returns:
            Number of papers successfully queued
        """
        queued_count = 0

        with self.db.get_session() as session:
            for paper_id in paper_ids:
                # Get paper
                statement = select(Paper).where(Paper.id == paper_id)
                paper = session.exec(statement).first()

                if paper and paper.status == "analyzed":
                    paper.status = "queued"
                    session.add(paper)
                    queued_count += 1

            session.commit()

        logger.info(f"Queued {queued_count}/{len(paper_ids)} papers for processing")
        return queued_count

    def reject_papers(self, project_id: str, paper_ids: List[int]) -> int:
        """
        Reject selected papers (delete from database).

        Args:
            project_id: Project ID (not used - each project has own DB)
            paper_ids: List of paper IDs to reject

        Returns:
            Number of papers successfully rejected
        """
        rejected_count = 0

        with self.db.get_session() as session:
            for paper_id in paper_ids:
                # Get paper
                statement = select(Paper).where(Paper.id == paper_id)
                paper = session.exec(statement).first()

                if paper and paper.status in ["detected", "analyzed"]:
                    # Delete paper record (file will be moved to _FAILED by caller)
                    session.delete(paper)
                    rejected_count += 1

            session.commit()

        logger.info(f"Rejected {rejected_count}/{len(paper_ids)} papers")
        return rejected_count

    def get_queue_status(self, project_id: str) -> Dict[str, Any]:
        """
        Get current queue processing status.

        Args:
            project_id: Project ID (not used - each project has own DB)

        Returns:
            Dictionary with queue statistics
        """
        with self.db.get_session() as session:
            # Count papers by status
            pending_count = session.exec(
                select(Paper).where(Paper.status == "analyzed")
            ).all()

            queued_count = session.exec(
                select(Paper).where(Paper.status == "queued")
            ).all()

            processing_paper = session.exec(
                select(Paper).where(
                    Paper.status == "processing"
                ).limit(1)
            ).first()

            return {
                "pending_count": len(pending_count),
                "queued_count": len(queued_count),
                "current_processing": {
                    "id": processing_paper.id,
                    "title": processing_paper.title,
                    "filename": Path(processing_paper.file_path).name
                } if processing_paper else None
            }

    def update_paper_status(self, paper_id: int, new_status: str) -> bool:
        """
        Update paper status.

        Args:
            paper_id: Paper ID
            new_status: New status value

        Returns:
            True if updated successfully
        """
        with self.db.get_session() as session:
            statement = select(Paper).where(Paper.id == paper_id)
            paper = session.exec(statement).first()

            if paper:
                paper.status = new_status
                session.add(paper)
                session.commit()
                logger.info(f"Updated paper {paper_id} status to '{new_status}'")
                return True

            return False
