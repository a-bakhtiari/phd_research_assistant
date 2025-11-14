"""
Project management service.

Handles CRUD operations for research projects.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from src.core.database import DatabaseManager
from src.core.vector_store import VectorStoreManager
from src.models.schemas import ProjectResponse, ProjectStats
from src.config import get_settings

logger = logging.getLogger(__name__)


class ProjectService:
    """Service for managing research projects."""

    def __init__(self, projects_root: Path):
        """
        Initialize project service.

        Args:
            projects_root: Root directory containing all projects
        """
        self.projects_root = projects_root
        self.projects_root.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ProjectService with root: {projects_root}")

    def get_all_projects(self) -> List[ProjectResponse]:
        """
        Get all existing projects.

        Returns:
            List of ProjectResponse objects
        """
        projects = []

        # Scan projects directory
        if not self.projects_root.exists():
            return projects

        for project_dir in self.projects_root.iterdir():
            if project_dir.is_dir() and not project_dir.name.startswith('.'):
                try:
                    # Check if it has a database
                    db_path = project_dir / "data" / "project.db"
                    if db_path.exists():
                        # Get paper count
                        db_manager = DatabaseManager(str(db_path))
                        papers = db_manager.get_all_papers()
                        paper_count = len(papers)

                        projects.append(ProjectResponse(
                            id=project_dir.name,
                            name=project_dir.name,
                            description=f"Research project: {project_dir.name}",
                            path=str(project_dir),
                            created_at=datetime.fromtimestamp(project_dir.stat().st_ctime),
                            paper_count=paper_count
                        ))
                except Exception as e:
                    logger.warning(f"Error reading project {project_dir.name}: {e}")
                    continue

        return projects

    def get_project(self, project_id: str) -> Optional[ProjectResponse]:
        """
        Get a specific project.

        Args:
            project_id: Project identifier (directory name)

        Returns:
            ProjectResponse or None if not found
        """
        project_dir = self.projects_root / project_id

        if not project_dir.exists() or not project_dir.is_dir():
            return None

        db_path = project_dir / "data" / "project.db"
        if not db_path.exists():
            return None

        try:
            db_manager = DatabaseManager(str(db_path))
            papers = db_manager.get_all_papers()

            return ProjectResponse(
                id=project_id,
                name=project_id,
                description=f"Research project: {project_id}",
                path=str(project_dir),
                created_at=datetime.fromtimestamp(project_dir.stat().st_ctime),
                paper_count=len(papers)
            )
        except Exception as e:
            logger.error(f"Error getting project {project_id}: {e}")
            return None

    def create_project(self, name: str, description: Optional[str] = None) -> ProjectResponse:
        """
        Create a new research project.

        Args:
            name: Project name (will be sanitized for filesystem)
            description: Optional project description

        Returns:
            Created ProjectResponse
        """
        # Sanitize project name
        safe_name = name.replace(" ", "_").replace("/", "_")
        project_dir = self.projects_root / safe_name

        if project_dir.exists():
            raise ValueError(f"Project '{safe_name}' already exists")

        # Create project structure
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "data").mkdir(exist_ok=True)
        (project_dir / "data" / "PAPERS").mkdir(exist_ok=True)
        (project_dir / "data" / "_NEW_PAPERS").mkdir(exist_ok=True)
        (project_dir / "data" / "_PROCESSING").mkdir(exist_ok=True)
        (project_dir / "data" / "_FAILED").mkdir(exist_ok=True)
        (project_dir / "tmp").mkdir(exist_ok=True)
        (project_dir / "documents").mkdir(exist_ok=True)
        (project_dir / "chat_sessions").mkdir(exist_ok=True)

        # Initialize database
        db_path = project_dir / "data" / "project.db"
        db_manager = DatabaseManager(str(db_path))

        # Initialize vector store with embedding config
        vector_store_path = project_dir / "data" / "vector_store"
        settings = get_settings()
        embedding_config = {
            "api_key": settings.openai_api_key,
            "model": settings.embedding_model
        }
        vector_manager = VectorStoreManager(str(vector_store_path), embedding_config=embedding_config)

        logger.info(f"Created new project: {safe_name}")

        return ProjectResponse(
            id=safe_name,
            name=name,
            description=description or f"Research project: {name}",
            path=str(project_dir),
            created_at=datetime.now(),
            paper_count=0
        )

    def delete_project(self, project_id: str) -> bool:
        """
        Delete a project.

        Args:
            project_id: Project identifier

        Returns:
            True if deleted, False if not found
        """
        project_dir = self.projects_root / project_id

        if not project_dir.exists():
            return False

        try:
            import shutil
            shutil.rmtree(project_dir)
            logger.info(f"Deleted project: {project_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            raise

    def get_project_stats(self, project_id: str) -> Optional[ProjectStats]:
        """
        Get statistics for a project.

        Args:
            project_id: Project identifier

        Returns:
            ProjectStats or None if project not found
        """
        project_dir = self.projects_root / project_id
        db_path = project_dir / "data" / "project.db"

        if not db_path.exists():
            return None

        try:
            db_manager = DatabaseManager(str(db_path))
            all_papers = db_manager.get_all_papers()

            # Count by status
            papers_by_status = {}
            for paper in all_papers:
                status = paper.status or "unread"
                papers_by_status[status] = papers_by_status.get(status, 0) + 1

            # Count by year
            papers_by_year = {}
            for paper in all_papers:
                if paper.year:
                    year_str = str(paper.year)
                    papers_by_year[year_str] = papers_by_year.get(year_str, 0) + 1

            # Count chat sessions
            chat_sessions_dir = project_dir / "chat_sessions"
            chat_session_count = len(list(chat_sessions_dir.glob("*.json"))) if chat_sessions_dir.exists() else 0

            # Count vector embeddings
            vector_manager = VectorStoreManager(str(project_dir / "data" / "vector_store"))
            vector_count = vector_manager.collection.count()

            return ProjectStats(
                total_papers=len(all_papers),
                papers_by_status=papers_by_status,
                papers_by_year=papers_by_year,
                total_chat_sessions=chat_session_count,
                total_vector_embeddings=vector_count
            )
        except Exception as e:
            logger.error(f"Error getting stats for project {project_id}: {e}")
            return None

    def get_db_manager(self, project_id: str) -> DatabaseManager:
        """
        Get DatabaseManager for a project.

        Args:
            project_id: Project identifier

        Returns:
            DatabaseManager instance
        """
        db_path = self.projects_root / project_id / "data" / "project.db"
        return DatabaseManager(str(db_path))

    def get_vector_manager(self, project_id: str) -> VectorStoreManager:
        """
        Get VectorStoreManager for a project.

        Args:
            project_id: Project identifier

        Returns:
            VectorStoreManager instance
        """
        settings = get_settings()
        vector_store_path = self.projects_root / project_id / "data" / "vector_store"

        # Build embedding configuration (OpenAI only)
        embedding_config = {
            "model": settings.embedding_model,
            "api_key": settings.openai_api_key,
            "batch_size": settings.embedding_batch_size,
            "max_retries": settings.embedding_max_retries
        }

        # Add cache configuration if enabled
        if settings.embedding_cache_enabled:
            embedding_config["cache"] = {
                "enabled": True,
                "path": settings.embedding_cache_path
            }

        return VectorStoreManager(
            str(vector_store_path),
            embedding_config=embedding_config
        )
