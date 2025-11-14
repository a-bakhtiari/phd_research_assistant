"""
Pydantic models for API requests and responses.

These models define the shape of data exchanged between frontend and backend.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================================================
# Project Models
# ============================================================================

class ProjectCreate(BaseModel):
    """Request model for creating a new project."""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = None


class ProjectResponse(BaseModel):
    """Response model for project data."""
    id: str
    name: str
    description: Optional[str]
    path: str
    created_at: Optional[datetime] = None
    paper_count: int = 0

    class Config:
        from_attributes = True


class ProjectStats(BaseModel):
    """Statistics for a project."""
    total_papers: int
    papers_by_status: Dict[str, int]
    papers_by_year: Dict[str, int]
    total_chat_sessions: int
    total_vector_embeddings: int


# ============================================================================
# Paper Models
# ============================================================================

class PaperResponse(BaseModel):
    """Response model for paper data."""
    id: int
    filename: str
    original_filename: str
    title: str
    authors: List[str]
    year: Optional[int]
    summary: Optional[str]
    abstract: Optional[str]
    status: str
    tags: List[str]
    date_added: datetime
    file_path: str
    doi: Optional[str]
    journal: Optional[str]
    page_count: Optional[int]
    citation_count: Optional[int]
    venue: Optional[str]
    semantic_scholar_url: Optional[str]

    # Large PDF handling fields
    needs_confirmation: bool = False
    confirmation_metadata: Optional[Dict[str, Any]] = None
    # confirmation_metadata contains:
    # - page_count: int (original page count)
    # - page_count_after_refs: int (after reference removal)
    # - estimated_time_minutes: float
    # - estimated_cost_usd: float
    # - references_removed: bool

    class Config:
        from_attributes = True


class PaperUpdate(BaseModel):
    """Request model for updating paper metadata."""
    status: Optional[str] = None
    tags: Optional[List[str]] = None
    summary: Optional[str] = None


class PaperListResponse(BaseModel):
    """Response model for list of papers."""
    papers: List[PaperResponse]
    total: int
    page: int
    page_size: int


# ============================================================================
# Chat Models
# ============================================================================

class ChatSessionCreate(BaseModel):
    """Request model for creating a chat session."""
    project_id: str
    title: Optional[str] = "New Chat Session"


class SourceCitation(BaseModel):
    """Citation information for a source."""
    paper_id: Optional[int] = None
    paper_title: str
    authors: Optional[List[str]] = []
    year: Optional[int] = None
    content: Optional[str] = ""
    page_number: Optional[int] = None
    similarity_score: Optional[float] = 0.0


class ChatMessageRequest(BaseModel):
    """Request model for sending a chat message."""
    message: str = Field(..., min_length=1)
    max_sources: int = Field(default=5, ge=1, le=20)
    use_rag: bool = Field(default=True, description="Whether to use RAG (search papers) or direct LLM")


class ChatMessageResponse(BaseModel):
    """Response model for chat message."""
    message_id: str
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    sources: Optional[List[SourceCitation]] = None
    tokens_used: Optional[Dict[str, int]] = None
    confidence_score: Optional[float] = None


class ChatSessionResponse(BaseModel):
    """Response model for chat session."""
    session_id: str
    project_id: str
    title: str
    created_at: datetime
    last_updated: datetime
    message_count: int
    total_tokens_used: int


# ============================================================================
# Recommendation Models
# ============================================================================

class RecommendationQueryRequest(BaseModel):
    """Request model for getting recommendations."""
    project_id: str
    query: str = Field(..., min_length=1)
    max_recommendations: int = Field(default=10, ge=1, le=50)


class PaperRecommendationItem(BaseModel):
    """A single paper recommendation."""
    title: str
    authors: List[str]
    year: int
    summary: str
    relevance_score: float
    reason: str
    doi: Optional[str]
    url: Optional[str]
    semantic_scholar_id: Optional[str]
    citation_count: Optional[int]
    venue: Optional[str]


class RecommendationResponse(BaseModel):
    """Response model for paper recommendations."""
    query: str
    recommendations: List[PaperRecommendationItem]
    gap_analysis: Optional[str] = None
    timestamp: datetime


# ============================================================================
# Agent Models
# ============================================================================

class AgentTaskRequest(BaseModel):
    """Request model for starting an agent task."""
    project_id: str
    task_type: str  # "discovery", "recommendation"
    parameters: Dict[str, Any]


class AgentTaskStatus(BaseModel):
    """Status information for an agent task."""
    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(ge=0.0, le=1.0)
    current_step: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class AgentTaskResponse(BaseModel):
    """Response model for agent task."""
    task_id: str
    task_type: str
    status: AgentTaskStatus
    results: Optional[Dict[str, Any]] = None


# ============================================================================
# Settings Models
# ============================================================================

class EmbeddingModelInfo(BaseModel):
    """Information about an embedding model."""
    name: str
    description: str
    dimension: int
    is_current: bool


class LLMSettingsResponse(BaseModel):
    """Current LLM settings."""
    default_provider: str
    available_providers: List[str]
    models: Dict[str, str]


class LLMSettingsUpdate(BaseModel):
    """Request model for updating LLM settings."""
    default_provider: Optional[str] = None
    model: Optional[str] = None


# ============================================================================
# Document Analysis Models
# ============================================================================

class DocumentInfo(BaseModel):
    """Information about a document file."""
    filename: str
    file_path: str
    size_bytes: int
    last_modified: datetime


class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis."""
    title: str
    word_count: int
    research_questions: List[str]
    key_concepts: List[str]
    literature_gaps: List[str]
    citations_present: List[str]
    citations_needed: List[str]
    document_type: str
    confidence_score: float


class DocumentAnalysisRequest(BaseModel):
    """Request model for analyzing documents."""
    project_id: str
    document_paths: List[str] = Field(..., min_items=1)


class DocumentRecommendationRequest(BaseModel):
    """Request model for document-based recommendations."""
    project_id: str
    document_paths: List[str] = Field(..., min_items=1)
    max_recommendations: int = Field(default=10, ge=1, le=50)


# ============================================================================
# Queue Models
# ============================================================================

class QueueStatusResponse(BaseModel):
    """Response model for queue status."""
    pending_count: int
    queued_count: int
    current_processing: Optional[Dict[str, Any]] = None


class BatchSelectRequest(BaseModel):
    """Request model for batch selecting/rejecting papers."""
    project_id: str  # Project identifier (e.g., 'test_proj')
    paper_ids: List[int]


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str
