from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, create_engine, Session, select, text
from pathlib import Path
import json
from datetime import datetime

from .models import Paper, ProjectConfig, ResearchSession, Citation


class DatabaseManager:
    """Manages database operations for the research assistant."""
    
    def __init__(self, db_path: str):
        """Initialize database manager with SQLite database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create database engine
        self.engine = create_engine(f"sqlite:///{self.db_path}")
        
        # Create tables
        SQLModel.metadata.create_all(self.engine)
        
        # Run database migrations
        self._run_migrations()
    
    def get_session(self) -> Session:
        """Get database session."""
        return Session(self.engine)
    
    # Paper operations
    def add_paper(self, paper_data: Dict[str, Any]) -> Paper:
        """Add a new paper to the database."""
        with self.get_session() as session:
            paper = Paper(**paper_data)
            session.add(paper)
            session.commit()
            session.refresh(paper)
            return paper
    
    def get_paper_by_id(self, paper_id: int) -> Optional[Paper]:
        """Get paper by ID."""
        with self.get_session() as session:
            return session.get(Paper, paper_id)
    
    def get_paper_by_filename(self, filename: str) -> Optional[Paper]:
        """Get paper by filename."""
        with self.get_session() as session:
            statement = select(Paper).where(Paper.filename == filename)
            return session.exec(statement).first()
    
    def get_paper_by_original_filename(self, original_filename: str) -> Optional[Paper]:
        """Get paper by original filename."""
        with self.get_session() as session:
            statement = select(Paper).where(Paper.original_filename == original_filename)
            return session.exec(statement).first()
    
    def get_papers_by_status(self, status: str) -> List[Paper]:
        """Get papers by reading status."""
        with self.get_session() as session:
            statement = select(Paper).where(Paper.status == status)
            return list(session.exec(statement))
    
    def get_papers_by_year_range(self, start_year: int, end_year: int) -> List[Paper]:
        """Get papers within a year range."""
        with self.get_session() as session:
            statement = select(Paper).where(
                Paper.year >= start_year,
                Paper.year <= end_year
            )
            return list(session.exec(statement))
    
    def search_papers_by_title(self, search_term: str) -> List[Paper]:
        """Search papers by title (case-insensitive)."""
        with self.get_session() as session:
            statement = select(Paper).where(
                Paper.title.ilike(f"%{search_term}%")
            )
            return list(session.exec(statement))
    
    def search_papers_by_author(self, author_name: str) -> List[Paper]:
        """Search papers by author name."""
        with self.get_session() as session:
            statement = select(Paper).where(
                Paper.authors.ilike(f"%{author_name}%")
            )
            return list(session.exec(statement))
    
    def update_paper_status(self, paper_id: int, status: str) -> bool:
        """Update paper reading status."""
        with self.get_session() as session:
            paper = session.get(Paper, paper_id)
            if paper:
                paper.status = status
                session.add(paper)
                session.commit()
                return True
            return False
    
    def update_paper_tags(self, paper_id: int, tags: List[str]) -> bool:
        """Update paper tags."""
        with self.get_session() as session:
            paper = session.get(Paper, paper_id)
            if paper:
                paper.set_tags_list(tags)
                session.add(paper)
                session.commit()
                return True
            return False
    
    def update_paper(self, paper_id: int, updates: Dict[str, Any]) -> bool:
        """Update paper with arbitrary fields."""
        with self.get_session() as session:
            paper = session.get(Paper, paper_id)
            if paper:
                for field, value in updates.items():
                    if hasattr(paper, field):
                        setattr(paper, field, value)
                session.add(paper)
                session.commit()
                return True
            return False
    
    def get_all_papers(self, limit: Optional[int] = None) -> List[Paper]:
        """Get all papers, optionally limited."""
        with self.get_session() as session:
            statement = select(Paper).order_by(Paper.date_added.desc())
            if limit:
                statement = statement.limit(limit)
            return list(session.exec(statement))
    
    def delete_paper(self, paper_id: int) -> bool:
        """Delete a paper from the database."""
        with self.get_session() as session:
            paper = session.get(Paper, paper_id)
            if paper:
                session.delete(paper)
                session.commit()
                return True
            return False
    
    # Configuration operations
    def set_config(self, key: str, value: str, category: str = "general", 
                   description: str = None) -> ProjectConfig:
        """Set or update a configuration value."""
        with self.get_session() as session:
            # Check if config exists
            statement = select(ProjectConfig).where(ProjectConfig.key == key)
            config = session.exec(statement).first()
            
            if config:
                config.value = value
                config.updated_at = datetime.utcnow()
                if description:
                    config.description = description
            else:
                config = ProjectConfig(
                    key=key,
                    value=value,
                    category=category,
                    description=description
                )
            
            session.add(config)
            session.commit()
            session.refresh(config)
            return config
    
    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """Get configuration value by key."""
        with self.get_session() as session:
            statement = select(ProjectConfig).where(ProjectConfig.key == key)
            config = session.exec(statement).first()
            return config.value if config else default
    
    def get_configs_by_category(self, category: str) -> List[ProjectConfig]:
        """Get all configurations in a category."""
        with self.get_session() as session:
            statement = select(ProjectConfig).where(ProjectConfig.category == category)
            return list(session.exec(statement))
    
    # Research session operations
    def start_research_session(self, session_type: str) -> ResearchSession:
        """Start a new research session."""
        with self.get_session() as session:
            research_session = ResearchSession(session_type=session_type)
            session.add(research_session)
            session.commit()
            session.refresh(research_session)
            return research_session
    
    def end_research_session(self, session_id: int, notes: str = None, 
                           word_count: int = None) -> bool:
        """End a research session."""
        with self.get_session() as session:
            research_session = session.get(ResearchSession, session_id)
            if research_session:
                research_session.end_time = datetime.utcnow()
                if notes:
                    research_session.notes = notes
                if word_count:
                    research_session.word_count = word_count
                session.add(research_session)
                session.commit()
                return True
            return False
    
    # Statistics and analytics
    def get_paper_stats(self) -> Dict[str, int]:
        """Get basic paper statistics."""
        with self.get_session() as session:
            total_papers = len(session.exec(select(Paper)).all())
            read_papers = len(session.exec(select(Paper).where(Paper.status == "read")).all())
            reading_papers = len(session.exec(select(Paper).where(Paper.status == "reading")).all())
            unread_papers = len(session.exec(select(Paper).where(Paper.status == "unread")).all())
            
            return {
                "total": total_papers,
                "read": read_papers,
                "reading": reading_papers,
                "unread": unread_papers
            }
    
    def get_papers_by_year_stats(self) -> Dict[int, int]:
        """Get paper count by publication year."""
        with self.get_session() as session:
            papers = session.exec(select(Paper)).all()
            year_stats = {}
            for paper in papers:
                if paper.year:
                    year_stats[paper.year] = year_stats.get(paper.year, 0) + 1
            return year_stats
    
    def _run_migrations(self):
        """Run database migrations to add new columns."""
        try:
            # Check if AI summary columns exist
            with self.engine.connect() as connection:
                # Try to query one of the newer columns to see if latest migration is needed
                try:
                    result = connection.execute(text("SELECT ai_purpose_rationale_research_question FROM paper LIMIT 1"))
                    # If this succeeds, latest migration is not needed
                    print("Latest AI summary schema already exists")
                    return
                except Exception:
                    # New columns don't exist, run migration
                    pass
                
                # Add legacy AI summary columns (if they don't exist)
                legacy_migration_queries = [
                    "ALTER TABLE paper ADD COLUMN ai_executive_summary TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_context_and_problem TEXT", 
                    "ALTER TABLE paper ADD COLUMN ai_research_questions TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_methodology TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_key_findings TEXT", 
                    "ALTER TABLE paper ADD COLUMN ai_primary_contributions TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_summary_generated_at TIMESTAMP"
                ]
                
                # Add new AI summary columns  
                new_migration_queries = [
                    "ALTER TABLE paper ADD COLUMN ai_purpose_rationale_research_question TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_theory_framework TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_major_findings_contributions TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_study_limitations_gaps TEXT",
                    "ALTER TABLE paper ADD COLUMN ai_study_implications TEXT"
                ]
                
                all_queries = legacy_migration_queries + new_migration_queries
                
                for query in all_queries:
                    try:
                        connection.execute(text(query))
                        print(f"Migration executed: {query}")
                    except Exception as e:
                        # Column might already exist, continue
                        print(f"Migration query failed (possibly already exists): {query} - {e}")
                        continue
                
                connection.commit()
                print("Database migration completed successfully")
                
        except Exception as e:
            print(f"Migration error: {e}")
            # Don't fail the initialization if migration has issues