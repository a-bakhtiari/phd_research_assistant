from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, create_engine, Session
from pydantic import field_validator
import json


class Paper(SQLModel, table=True):
    """Database model for academic papers."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str = Field(unique=True, index=True)
    original_filename: str
    authors: str  # JSON string of author list
    year: Optional[int] = Field(default=None, index=True)
    title: str = Field(index=True)
    summary: Optional[str] = Field(default=None)
    status: str = Field(default="unread")  # unread, reading, read
    tags: Optional[str] = Field(default=None)  # JSON string of tags
    date_added: datetime = Field(default_factory=datetime.utcnow)
    file_path: str
    doi: Optional[str] = Field(default=None, index=True)
    journal: Optional[str] = Field(default=None)
    page_count: Optional[int] = Field(default=None)
    abstract: Optional[str] = Field(default=None)

    # Cleaned text (LLM-processed, no headers/footers/tables/figures/references)
    cleaned_full_text: Optional[str] = Field(default=None)

    # Semantic Scholar metadata fields
    citation_count: Optional[int] = Field(default=None, index=True)
    influential_citation_count: Optional[int] = Field(default=None) 
    venue: Optional[str] = Field(default=None)  # Journal/conference from Semantic Scholar
    is_open_access: Optional[bool] = Field(default=None)
    open_access_pdf: Optional[str] = Field(default=None)
    semantic_scholar_id: Optional[str] = Field(default=None, index=True)
    semantic_scholar_url: Optional[str] = Field(default=None)
    tldr_summary: Optional[str] = Field(default=None)
    fields_of_study: Optional[str] = Field(default=None)  # JSON string of research areas
    ss_last_updated: Optional[datetime] = Field(default=None)  # When SS data was last updated
    
    # AI-generated structured summary fields
    ai_executive_summary: Optional[str] = Field(default=None)
    ai_purpose_rationale_research_question: Optional[str] = Field(default=None)  # JSON string of purpose/rationale/RQ
    ai_theory_framework: Optional[str] = Field(default=None)  # JSON string of theoretical framework
    ai_methodology: Optional[str] = Field(default=None)  # JSON string of methodology components
    ai_major_findings_contributions: Optional[str] = Field(default=None)  # JSON string of findings and contributions
    ai_study_limitations_gaps: Optional[str] = Field(default=None)  # JSON string of limitations and gaps
    ai_study_implications: Optional[str] = Field(default=None)  # JSON string of research/practice/policy implications
    ai_summary_generated_at: Optional[datetime] = Field(default=None)  # When AI summary was generated
    
    # Legacy AI summary fields (kept for backward compatibility)
    ai_context_and_problem: Optional[str] = Field(default=None)  # JSON string of context/problem statements
    ai_research_questions: Optional[str] = Field(default=None)  # JSON string of research questions
    ai_key_findings: Optional[str] = Field(default=None)  # JSON string of key findings
    ai_primary_contributions: Optional[str] = Field(default=None)  # JSON string of contributions
    
    @field_validator("status")
    @classmethod
    def validate_status(cls, v):
        allowed_statuses = [
            # Queue workflow statuses
            "detected",              # File detected, awaiting quick analysis
            "analyzed",              # Quick analysis complete, awaiting user selection
            "queued",                # User selected for processing, in queue
            "processing",            # Currently being processed
            "pending_confirmation",  # Large PDF awaiting user confirmation
            # Reading workflow statuses
            "unread",                # Processed and ready to read
            "reading",               # Currently reading
            "read"                   # Finished reading
        ]
        if v not in allowed_statuses:
            raise ValueError(f"Status must be one of {allowed_statuses}")
        return v
    
    def get_authors_list(self) -> List[str]:
        """Parse authors JSON string into list."""
        try:
            return json.loads(self.authors) if self.authors else []
        except json.JSONDecodeError:
            return [self.authors] if self.authors else []
    
    def set_authors_list(self, authors_list: List[str]):
        """Convert authors list to JSON string."""
        self.authors = json.dumps(authors_list)
    
    def get_tags_list(self) -> List[str]:
        """Parse tags JSON string into list."""
        try:
            return json.loads(self.tags) if self.tags else []
        except json.JSONDecodeError:
            return [self.tags] if self.tags else []
    
    def set_tags_list(self, tags_list: List[str]):
        """Convert tags list to JSON string."""
        self.tags = json.dumps(tags_list)


class ProjectConfig(SQLModel, table=True):
    """Database model for project configuration settings."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True, index=True)
    value: str
    category: str = Field(default="general")
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ResearchSession(SQLModel, table=True):
    """Database model for tracking research sessions and activities."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    session_type: str  # writing, reading, reviewing, etc.
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(default=None)
    papers_read: Optional[str] = Field(default=None)  # JSON list of paper IDs
    notes: Optional[str] = Field(default=None)
    word_count: Optional[int] = Field(default=None)
    
    def get_papers_read_list(self) -> List[int]:
        """Parse papers_read JSON string into list."""
        try:
            return json.loads(self.papers_read) if self.papers_read else []
        except json.JSONDecodeError:
            return []
    
    def set_papers_read_list(self, paper_ids: List[int]):
        """Convert paper IDs list to JSON string."""
        self.papers_read = json.dumps(paper_ids)


class Citation(SQLModel, table=True):
    """Database model for managing citations and references."""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    paper_id: int = Field(foreign_key="paper.id")
    citation_key: str = Field(unique=True, index=True)  # BibTeX key
    bibtex_entry: str
    citation_style: str = Field(default="apa")
    custom_note: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)